#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "nano_graphrag/utils/Types.hpp"
#include "nano_graphrag/operations/chunking/base.hpp"

namespace nano_graphrag
{

/**
 * @brief Default chunking strategy based on token counts
 */
class DefaultChunkingStrategy : public IChunkingStrategy
{
public:
  /**
   * @brief Constructor
   */
  DefaultChunkingStrategy()
  {
    // Set sensible defaults
    chunk_size_ = 1024;
    overlap_size_ = 128;
  }

  /**
   * @brief Chunk the document into smaller chunks based on token size
   *
   * @param doc The input document
   * @return The chunks as a vector of strings
   */
  std::vector<std::string> chunk(const std::string& doc) const override
  {
    std::vector<int> tokens = tokenizer_->encode(doc);
    std::vector<std::vector<int>> tokens_list;
    tokens_list.push_back(std::move(tokens));
    std::vector<std::string> docs{ doc };
    std::vector<std::string> doc_keys{ std::string("doc") };
    auto chunks = chunking_by_token_size(tokens_list, docs, doc_keys, overlap_size_, chunk_size_);
    std::vector<std::string> out;
    out.reserve(chunks.size());
    for (const auto& ch : chunks)
      out.push_back(ch.content);
    return out;
  }

  /**
   * @brief Chunking by token size helper
   *
   * @param tokens_list List of tokenized documents
   * @param docs Original documents
   * @param doc_keys Document keys
   * @param overlap_token_size Overlap size between chunks
   * @param max_token_size Max chunk size
   * @return std::vector<TextChunk> The resulting text chunks
   */
  inline std::vector<TextChunk> chunking_by_token_size(std::vector<std::vector<int>>& tokens_list,
                                                       std::vector<std::string>& docs,
                                                       std::vector<std::string>& doc_keys,
                                                       int overlap_token_size = 128,
                                                       int max_token_size = 1024) const
  {
    std::vector<TextChunk> results;
    for (size_t index = 0; index < tokens_list.size(); ++index)
    {
      const auto& tokens = tokens_list[index];
      std::vector<std::vector<int>> chunk_token;
      std::vector<int> lengths;
      std::vector<int> starts;
      for (int start = 0; start < (int)tokens.size(); start += (max_token_size - overlap_token_size))
      {
        std::vector<int> sub(tokens.begin() + start,
                             tokens.begin() + std::min<int>(start + max_token_size, tokens.size()));
        chunk_token.push_back(std::move(sub));
        lengths.push_back(std::min(max_token_size, (int)tokens.size() - start));
        starts.push_back(start);
      }
        std::vector<std::string> chunk_texts =
          tokenizer_->decode(chunk_token, docs[index], starts, lengths);
      for (size_t i = 0; i < chunk_texts.size(); ++i)
      {
        results.push_back(TextChunk{ lengths[i], chunk_texts[i], doc_keys[index], (int)i });
      }
    }
    return results;
  }

  /**
   * @brief Get chunks from new documents
   *
   * @param new_docs The new documents as a map of doc_id to content
   * @param overlap_token_size Overlap size between chunks
   * @param max_token_size Max chunk size
   * @return std::unordered_map<std::string, TextChunk> The resulting text chunks mapped by chunk ID
   */
  inline std::unordered_map<std::string, TextChunk>
  get_chunks(const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& new_docs,
             int overlap_token_size = 128, int max_token_size = 1024)
  {
    std::unordered_map<std::string, TextChunk> inserting_chunks;
    std::vector<std::string> docs;
    docs.reserve(new_docs.size());
    std::vector<std::string> keys;
    keys.reserve(new_docs.size());
    for (const auto& kv : new_docs)
    {
      keys.push_back(kv.first);
      docs.push_back(kv.second.at("content"));
    }
    std::vector<std::vector<int>> tokens;
    tokens.reserve(docs.size());
    for (const auto& d : docs)
      tokens.push_back(tokenizer_->encode(d));
    auto chunks = chunking_by_token_size(tokens, docs, keys, overlap_token_size, max_token_size);
    for (const auto& chunk : chunks)
    {
      // simple md5-like id: use std::hash
      std::hash<std::string> h;
      std::string id = "chunk-" + std::to_string(h(chunk.content));
      inserting_chunks.emplace(id, chunk);
    }
    return inserting_chunks;
  }

};

}  // namespace nano_graphrag
