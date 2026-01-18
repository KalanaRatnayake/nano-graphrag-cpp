#pragma once
#include <string>
#include <vector>

namespace nano_graphrag
{

/**
 * @brief Enum for different tokenizer types
 *
 * @param Simple Simple tokenizer (whitespace-based)
 * @param OpenAI OpenAI tokenizer
 */
enum class TokenizerType
{
  Simple,
  OpenAI,
  Tiktoken
};

/**
 * @brief Abstract base class for tokenizer strategies
 */
class ITokenizerStrategy
{
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~ITokenizerStrategy() = default;

  /**
   * @brief Given a text, return its token IDs
   * @param text The input text
   * @return The token IDs as a vector of integers
   */
  virtual std::vector<int> encode(const std::string& text) const = 0;

  /**
   * @brief Given a list of token ID sequences, return their decoded texts
   * @param tokens_list The list of token ID sequences
   * @return The decoded texts as a vector of strings
   */
  virtual std::vector<std::string> decode_batch(const std::vector<std::vector<int>>& tokens_list) const = 0;

  /**
   * @brief Given chunked token ID sequences, reconstruct their texts based on original document
   * @param chunk_token The chunked token ID sequences
   * @param doc The original document
   * @param starts The start positions of each chunk in the original token sequence
   * @param lengths The lengths of each chunk
   * @return The decoded chunk texts as a vector of strings
   */
  virtual std::vector<std::string> decode(const std::vector<std::vector<int>>& chunk_token,
                                          const std::string& doc, const std::vector<int>& starts,
                                          const std::vector<int>& lengths) const = 0;

  /**
   * @brief Get the tokenizer type
   * @return The tokenizer type
   */
  virtual TokenizerType type() const = 0;
};

}  // namespace nano_graphrag
