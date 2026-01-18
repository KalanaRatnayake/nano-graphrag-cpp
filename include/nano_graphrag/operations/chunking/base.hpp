// Abstract base class for chunking strategies
#pragma once
#include <string>
#include <vector>
#include <memory>
#include "nano_graphrag/operations/tokenize/base.hpp"
namespace nano_graphrag
{

/**
 * @brief Abstract base class for chunking strategies
 */
class IChunkingStrategy
{
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~IChunkingStrategy() = default;

  /**
   * @brief Given a document, return its chunks
   * @param doc The input document
   * @return The chunks as a vector of strings
   */
  virtual std::vector<std::string> chunk(const std::string& doc) const = 0;

  /**
   * @brief Inject tokenizer strategy
   * @param tokenizer Shared pointer to tokenizer strategy
   */
  virtual void set_tokenizer(std::shared_ptr<ITokenizerStrategy> tokenizer)
  {
    tokenizer_ = std::move(tokenizer);
  }

  /**
   * @brief Set max chunk size and overlap size
   *
   * @param size The max chunk size
   */
  virtual void set_chunk_size(int size)
  {
    chunk_size_ = size;
  }

  /**
   * @brief Set overlap size between chunks
   * @param size The overlap size
   */
  virtual void set_overlap_size(int size)
  {
    overlap_size_ = size;
  }

protected:
  int chunk_size_;
  int overlap_size_;
  std::shared_ptr<ITokenizerStrategy> tokenizer_;
};

}  // namespace nano_graphrag
