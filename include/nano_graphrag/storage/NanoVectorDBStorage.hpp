#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>
#include <memory>

#include "NanoVectorDB.hpp"

#include "nano_graphrag/storage/base.hpp"
#include "nano_graphrag/utils/Log.hpp"

namespace nano_graphrag
{

/**
 * @brief Vector storage backed by nano-vectordb-cpp.
 *
 * Wraps `nano_vectordb::NanoVectorDB` to index embeddings and perform nearest
 * neighbor queries. Uses `embedding_func` to compute vectors from input
 * `content` fields. Optional config:
 * - `metric`: similarity metric (e.g., "cosine").
 * - `storage_file`: path to persisted index file.
 * - `query_better_than_threshold`: minimum similarity score to include results.
 *
 * Metadata fields specified in `meta_fields` are captured per id and returned
 * with query results alongside the similarity score.
 */
class NanoVectorDBStorage : public BaseVectorStorage
{
public:
  explicit NanoVectorDBStorage(const std::string& ns = "",
                               const std::unordered_map<std::string, std::string>& cfg = {},
                               const std::shared_ptr<IEmbeddingStrategy>& emb = nullptr)
  {
    this->namespace_name = ns;
    this->global_config = cfg;
    this->embedding_strategy = emb;
    debug_log("[NanoVectorDBStorage] init ns=", ns);
    cosine_better_than_threshold_ = parse_double(cfg, "query_better_than_threshold", 0.2);
    std::string metric = cfg.count("metric") ? cfg.at("metric") : std::string("cosine");
    std::string storage_file =
        cfg.count("storage_file") ? cfg.at("storage_file") : std::string("nano-vectordb.json");

    if (embedding_strategy && embedding_strategy->embedding_dim() > 0)
    {
      debug_log("[NanoVectorDBStorage] dim=", embedding_strategy->embedding_dim(),
                ", metric=", metric, ", file=", storage_file);
      db_ = std::make_unique<nano_vectordb::NanoVectorDB>(
          static_cast<int>(embedding_strategy->embedding_dim()), metric, storage_file);

      // Initialize optional storage backend strategy
      if (cfg.count("storage_backend"))
      {
        std::string backend = cfg.at("storage_backend");
        if (backend == "sqlite" || backend == "SQLite")
        {
          debug_log("[NanoVectorDBStorage] initialize storage: SQLite");
          db_->initialize_storage(::nano_vectordb::storage::SQLite, storage_file);
        }
        else if (backend == "file" || backend == "File")
        {
          debug_log("[NanoVectorDBStorage] initialize storage: File");
          db_->initialize_storage(::nano_vectordb::storage::File, storage_file);
        }
      }

      // Initialize optional metric strategy to match updates
      if (metric == "l2" || metric == "L2")
      {
        debug_log("[NanoVectorDBStorage] metric: L2");
        db_->initialize_metric(::nano_vectordb::metric::L2);
      }
      else
      {
        debug_log("[NanoVectorDBStorage] metric: Cosine");
        db_->initialize_metric(::nano_vectordb::metric::Cosine);
      }

      // Auto-save config
      auto_save_ = parse_bool(cfg, "auto_save", false);
      debug_log("[NanoVectorDBStorage] auto_save=", auto_save_ ? "true" : "false");
    }
  }

  /**
   * @brief Upsert id->record maps; records should contain `content` for embedding.
   */
  void
  upsert(const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& data) override
  {
    if (data.empty())
      return;
    debug_log("[NanoVectorDBStorage] upsert count=", data.size());
    std::vector<std::string> contents;
    contents.reserve(data.size());
    std::vector<std::string> ids;
    ids.reserve(data.size());
    for (const auto& kv : data)
    {
      ids.push_back(kv.first);
      auto itc = kv.second.find("content");
      contents.push_back(itc != kv.second.end() ? itc->second : std::string{});

      // capture meta fields
      std::unordered_map<std::string, std::string> meta;
      for (const auto& mf : meta_fields)
      {
        if (mf.second)
        {
          auto itm = kv.second.find(mf.first);
          if (itm != kv.second.end())
            meta[mf.first] = itm->second;
        }
      }
      metas_[kv.first] = std::move(meta);
    }
    std::vector<std::vector<float>> embeddings;

    if (embedding_strategy)
      embeddings = embedding_strategy->embed(contents);
    debug_log("[NanoVectorDBStorage] embeddings size=", embeddings.size());

    if (embeddings.size() != ids.size())
    {
      size_t dim = embedding_strategy ? embedding_strategy->embedding_dim() : 0;
      embeddings.assign(ids.size(), std::vector<float>(dim, 0.0f));
    }

    if (!db_ && embedding_strategy && embedding_strategy->embedding_dim() > 0)
    {
      std::string metric = global_config.count("metric") ? global_config.at("metric") : std::string("cosine");
      std::string storage_file = global_config.count("storage_file") ? global_config.at("storage_file") :
                                                                       std::string("nano-vectordb.json");
      db_ = std::make_unique<nano_vectordb::NanoVectorDB>(
          static_cast<int>(embedding_strategy->embedding_dim()), metric, storage_file);
      debug_log("[NanoVectorDBStorage] late init DB dim=", embedding_strategy->embedding_dim());
    }

    std::vector<nano_vectordb::Data> datas;
    datas.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i)
    {
      Eigen::VectorXf v(static_cast<int>(embeddings[i].size()));
      for (int j = 0; j < v.size(); ++j)
        v[j] = embeddings[i][j];
      datas.push_back({ ids[i], v });
    }
    if (db_)
    {
      db_->upsert(datas);
      if (auto_save_)
        db_->save();
      debug_log("[NanoVectorDBStorage] upsert completed");
    }
  }

  /**
   * @brief Query nearest neighbors for a raw text string.
   * @param query Input text to embed and search.
   * @param top_k Max results to return.
   * @return Rows with `id`, `similarity`, and any captured metadata.
   */
  std::vector<std::unordered_map<std::string, std::string>> query(const std::string& query,
                                                                  int top_k) override
  {
    debug_log("[NanoVectorDBStorage] query top_k=", top_k);
    std::vector<std::vector<float>> qembs;
    if (embedding_strategy)
      qembs = embedding_strategy->embed(std::vector<std::string>{ query });
    debug_log("[NanoVectorDBStorage] query embed done size=", qembs.size());
    std::vector<float> q;
    if (!qembs.empty())
      q = qembs[0];
    else
      q.assign(embedding_strategy ? embedding_strategy->embedding_dim() : 0, 0.0f);
    std::vector<std::unordered_map<std::string, std::string>> out;
    if (db_)
    {
      Eigen::VectorXf v(static_cast<int>(q.size()));
      for (int i = 0; i < v.size(); ++i)
        v[i] = q[i];
      std::optional<float> th = std::nullopt;
      if (cosine_better_than_threshold_ > 0.0)
        th = static_cast<float>(cosine_better_than_threshold_);
      auto results = db_->query(v, top_k, th);
      debug_log("[NanoVectorDBStorage] results=", results.size());
      out.reserve(results.size());
      for (const auto& r : results)
      {
        std::unordered_map<std::string, std::string> row = metas_[r.data.id];
        row["id"] = r.data.id;
        row["similarity"] = std::to_string(r.score);
        out.push_back(std::move(row));
      }
      return out;
    }
    return {};
  }

private:
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> metas_;
  double cosine_better_than_threshold_{ 0.2 };
  std::unique_ptr<nano_vectordb::NanoVectorDB> db_;
  bool auto_save_{ false };

  /**
   * @brief Parse a double from config with default.
   */
  static inline double parse_double(const std::unordered_map<std::string, std::string>& cfg,
                                    const std::string& key, double def)
  {
    auto it = cfg.find(key);
    if (it == cfg.end())
      return def;
    try
    {
      return std::stod(it->second);
    }
    catch (...)
    {
      return def;
    }
  }

  static inline bool parse_bool(const std::unordered_map<std::string, std::string>& cfg,
                                const std::string& key, bool def)
  {
    auto it = cfg.find(key);
    if (it == cfg.end())
      return def;
    std::string v = it->second;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "1" || v == "true" || v == "yes");
  }
};

}  // namespace nano_graphrag
