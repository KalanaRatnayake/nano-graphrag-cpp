#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include "nano_graphrag/embedding/base.hpp"
#include "nano_graphrag/embedding/factory.hpp"
#include "nano_graphrag/operations/tokenize/base.hpp"
#include "nano_graphrag/operations/tokenize/factory.hpp"
#include "nano_graphrag/operations/chunking/default.hpp"
#include "nano_graphrag/operations/chunking/factory.hpp"
#include "nano_graphrag/storage/base.hpp"
#include "nano_graphrag/storage/JsonKVStorage.hpp"
#include "nano_graphrag/storage/GraphStorage.hpp"
#include "nano_graphrag/storage/NanoVectorDBStorage.hpp"
#include "nano_graphrag/llm/base.hpp"
#include "nano_graphrag/llm/factory.hpp"
#include "nano_graphrag/utils/Prompts.hpp"
#include "nano_graphrag/utils/Types.hpp"
#include "nano_graphrag/utils/Log.hpp"

namespace nano_graphrag
{

class GraphRAG
{
public:
  // config
  std::string working_dir;
  bool enable_local{ true };
  bool enable_naive_rag{ false };

  // chunking/tokenizer
  int chunk_token_size{ 1200 };
  int chunk_overlap_token_size{ 100 };
  std::shared_ptr<ITokenizerStrategy> tokenizer;
  std::unique_ptr<DefaultChunkingStrategy> chunker;

  // storage
  std::unique_ptr<BaseKVStorage<std::unordered_map<std::string, std::string>>> full_docs;
  std::unique_ptr<BaseKVStorage<TextChunk>> text_chunks;
  std::unique_ptr<BaseKVStorage<Community>> community_reports;
  std::unique_ptr<BaseGraphStorage> chunk_entity_relation_graph;
  std::unique_ptr<BaseVectorStorage> entities_vdb;
  std::unique_ptr<BaseVectorStorage> chunks_vdb;

  // strategies
  std::shared_ptr<IEmbeddingStrategy> embedding_strategy;  // to be set by user
  std::shared_ptr<ILLMStrategy> llm_strategy;              // to be set by user (defaults available)
  std::string chat_model{ "gpt-4.1" };

  explicit GraphRAG(const std::string& workdir = std::string{ "./nano_graphrag_cache" })
    : working_dir(workdir)
  {
    debug_log("[GraphRAG] init working_dir=", working_dir);
    std::filesystem::create_directories(working_dir);
    std::unordered_map<std::string, std::string> cfg;
    cfg["working_dir"] = working_dir;

    full_docs =
        std::make_unique<JsonKVStorage<std::unordered_map<std::string, std::string>>>("full_docs", cfg);
    text_chunks = std::make_unique<JsonKVStorage<TextChunk>>("text_chunks", cfg);
    community_reports = std::make_unique<JsonKVStorage<Community>>("community_reports", cfg);
    chunk_entity_relation_graph = std::make_unique<InMemoryGraphStorage>("chunk_entity_relation", cfg);

    // Defaults: Tiktoken tokenizer if available, else Simple
    tokenizer = create_tokenizer_strategy(TokenizerType::Tiktoken);
    if (!tokenizer)
      tokenizer = create_tokenizer_strategy(TokenizerType::Simple);
    chunker = std::make_unique<DefaultChunkingStrategy>();
    chunker->set_tokenizer(tokenizer);
    chunker->set_chunk_size(chunk_token_size);
    chunker->set_overlap_size(chunk_overlap_token_size);
    debug_log("[GraphRAG] tokenizer set, chunk_size=", chunk_token_size,
              ", overlap=", chunk_overlap_token_size);
  }

  void set_embedding_strategy(std::shared_ptr<IEmbeddingStrategy> s)
  {
    embedding_strategy = std::move(s);
  }
  void set_llm_strategy(std::shared_ptr<ILLMStrategy> s)
  {
    llm_strategy = std::move(s);
  }
  void set_chat_model(const std::string& m)
  {
    chat_model = m;
  }
  void set_tokenizer(TokenizerType type)
  {
    tokenizer = create_tokenizer_strategy(type);
    if (chunker)
      chunker->set_tokenizer(tokenizer);
  }
  void set_chunk_params(int max_tokens, int overlap_tokens)
  {
    chunk_token_size = max_tokens;
    chunk_overlap_token_size = overlap_tokens;
    if (chunker)
    {
      chunker->set_chunk_size(chunk_token_size);
      chunker->set_overlap_size(chunk_overlap_token_size);
    }
  }

  void enable_naive(bool v)
  {
    enable_naive_rag = v;
    if (enable_naive_rag)
    {
      debug_log("[GraphRAG] enabling naive mode");
      std::unordered_map<std::string, std::string> cfg;
      cfg["working_dir"] = working_dir;
      cfg["query_better_than_threshold"] = "0.0";
      chunks_vdb = std::make_unique<NanoVectorDBStorage>("chunks", cfg, embedding_strategy);
    }
  }

  void insert(const std::vector<std::string>& docs)
  {
    debug_log("[GraphRAG] insert docs count=", docs.size());
    // compute new doc ids
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> new_docs;
    std::hash<std::string> h;
    for (const auto& c : docs)
    {
      std::string id = "doc-" + std::to_string(h(c));
      new_docs[id] = { { "content", c } };
    }

    // chunking using DefaultChunkingStrategy helper
    std::unordered_map<std::string, TextChunk> inserting_chunks;
    if (chunker)
    {
      inserting_chunks = chunker->get_chunks(new_docs, chunk_overlap_token_size, chunk_token_size);
    }
    debug_log("[GraphRAG] chunks produced=", inserting_chunks.size());

    // upsert vector DB for naive
    if (enable_naive_rag && chunks_vdb)
    {
      debug_log("[GraphRAG] upserting chunks into VDB");
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>> vdb_data;
      for (const auto& kv : inserting_chunks)
      {
        vdb_data[kv.first] = { { "content", kv.second.content } };
      }
      chunks_vdb->upsert(vdb_data);
    }

    // upsert KV stores
    full_docs->upsert(new_docs);
    text_chunks->upsert(inserting_chunks);
    debug_log("[GraphRAG] insert completed");
  }

  std::string query(const std::string& q, const QueryParam& param = QueryParam{})
  {
    if (param.mode == "naive")
    {
      return naive_query(q, param);
    }
    // local/global not implemented fully
    return std::string{ "Sorry, I'm not able to provide an answer to that question." };
  }

private:
  std::string naive_query(const std::string& q, const QueryParam& param)
  {
    debug_log("[GraphRAG] naive_query top_k=", param.top_k,
              ", only_context=", param.only_need_context ? "true" : "false");
    if (!chunks_vdb)
      return std::string{ "Sorry, I'm not able to provide an answer to that question." };
    auto results = chunks_vdb->query(q, param.top_k);
    debug_log("[GraphRAG] VDB results=", results.size());
    if (results.empty())
      return std::string{ "Sorry, I'm not able to provide an answer to that question." };
    std::vector<std::string> ids;
    ids.reserve(results.size());
    for (auto& r : results)
      ids.push_back(r["id"]);
    auto chunks = text_chunks->get_by_ids(ids);
    std::string section;
    int tokens = 0;
    for (const auto& optChunk : chunks)
    {
      if (!optChunk.has_value())
        continue;
      const auto& c = optChunk.value();
      tokens += c.tokens;
      if (tokens > param.naive_max_token_for_text_unit)
        break;
      section += c.content + "\n--New Chunk--\n";
    }
    if (!section.empty())
      section.erase(section.size() - std::string("\n--New Chunk--\n").size());
    debug_log("[GraphRAG] context tokens=", tokens);
    if (param.only_need_context)
      return section;
    auto sys_prompt = Prompts::naive_rag_response(section, param.response_type);
    if (!llm_strategy)
      return section;
    debug_log("[GraphRAG] calling LLM");
    auto resp = llm_strategy->prompt(q, sys_prompt);
    return resp.empty() ? section : resp;
  }
};

}  // namespace nano_graphrag
