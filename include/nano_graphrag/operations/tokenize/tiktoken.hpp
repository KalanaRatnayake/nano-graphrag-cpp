#pragma once
#include <string>
#include <vector>
#include <memory>
#include "nano_graphrag/operations/tokenize/base.hpp"
#include "encoding.h"
#include "modelparams.h"

namespace nano_graphrag
{

/**
 * @brief Tokenizer using cpp-tiktoken (BPE-based) encodings
 */
class TiktokenTokenizer : public ITokenizerStrategy
{
public:
  /**
   * @brief Construct with default O200K_BASE encoding
   */
  TiktokenTokenizer()
  {
    encoder_ = GptEncoding::get_encoding(LanguageModel::O200K_BASE);
  }

  /**
   * @brief Construct with specified language model encoding
   * @param model The LanguageModel (e.g., O200K_BASE)
   */
  explicit TiktokenTokenizer(LanguageModel model)
  {
    encoder_ = GptEncoding::get_encoding(model);
  }

  /**
   * @brief Given a text, return its token IDs
   */
  std::vector<int> encode(const std::string& text) const override
  {
    return encoder_->encode(text);
  }

  /**
   * @brief Decode a batch of token sequences to strings
   */
  std::vector<std::string> decode_batch(const std::vector<std::vector<int>>& tokens_list) const override
  {
    std::vector<std::string> out;
    out.reserve(tokens_list.size());
    for (const auto& tk : tokens_list)
      out.push_back(encoder_->decode(tk));
    return out;
  }

  /**
   * @brief Decode chunked token sequences; doc/starts/lengths are not needed since chunks are already
   * token-sliced
   */
  std::vector<std::string> decode(const std::vector<std::vector<int>>& chunk_token,
                                  const std::string& /*doc*/, const std::vector<int>& /*starts*/,
                                  const std::vector<int>& /*lengths*/) const override
  {
    return decode_batch(chunk_token);
  }

  /**
   * @brief Get the tokenizer type
   */
  TokenizerType type() const override
  {
    return TokenizerType::Tiktoken;
  }

private:
  std::shared_ptr<GptEncoding> encoder_;
};

}  // namespace nano_graphrag
