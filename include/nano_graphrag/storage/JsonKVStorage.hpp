#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <optional>

#include "nano_graphrag/storage/base.hpp"

namespace nano_graphrag
{

/**
 * @brief Simple in-memory JSON-like key-value storage.
 *
 * Stores typed values `T` keyed by string ids. This backend is non-persistent
 * (lives in memory) and intended for prototyping or as a cache layer for
 * documents, chunks, and community reports.
 */
template <typename T>
class JsonKVStorage : public BaseKVStorage<T>
{
public:
  using Map = std::unordered_map<std::string, T>;

  explicit JsonKVStorage(const std::string& ns = "",
                         const std::unordered_map<std::string, std::string>& cfg = {})
  {
    this->namespace_name = ns;
    this->global_config = cfg;
  }

  /**
   * @brief Return all keys stored.
   */
  std::vector<std::string> all_keys() override
  {
    std::vector<std::string> keys;
    keys.reserve(data_.size());
    for (auto& kv : data_)
      keys.push_back(kv.first);
    return keys;
  }

  /**
   * @brief Get a single value by id.
   */
  std::optional<T> get_by_id(const std::string& id) override
  {
    auto it = data_.find(id);
    if (it == data_.end())
      return std::nullopt;
    return it->second;
  }

  /**
   * @brief Batch get values by ids.
   */
  std::vector<std::optional<T>> get_by_ids(const std::vector<std::string>& ids) override
  {
    std::vector<std::optional<T>> out;
    out.reserve(ids.size());
    for (auto& id : ids)
      out.push_back(get_by_id(id));
    return out;
  }

  /**
   * @brief Return ids that are missing from the store.
   */
  std::vector<std::string> filter_keys(const std::vector<std::string>& ids) override
  {
    std::vector<std::string> missing;
    for (auto& id : ids)
      if (data_.find(id) == data_.end())
        missing.push_back(id);
    return missing;
  }

  /**
   * @brief Upsert batch id->value pairs.
   */
  void upsert(const std::unordered_map<std::string, T>& data) override
  {
    for (auto& kv : data)
      data_[kv.first] = kv.second;
  }

  /**
   * @brief Clear all stored data.
   */
  void drop() override
  {
    data_.clear();
  }

private:
  Map data_{};
};

}  // namespace nano_graphrag
