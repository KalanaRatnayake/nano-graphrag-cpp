#pragma once
#include <string>
#include <vector>
#include "nano_graphrag/operations/tokenize/base.hpp"

namespace nano_graphrag
{

/**
 * @brief Simple tokenizer strategy (whitespace-based)
 */
class SimpleTokenizer : public ITokenizerStrategy
{
public:
  /**
   * @brief Given a text, return its token IDs
   * @param text The input text
   * @return The token IDs as a vector of integers
   */
  std::vector<int> encode(const std::string& text) const override
  {
    std::vector<int> tokens;
    tokens.reserve(text.size() / 4);
    bool in_word = false;
    for (char c : text)
    {
      if (c == ' ' || c == '\n' || c == '\t')
      {
        if (in_word)
        {
          tokens.push_back(1);
          in_word = false;
        }
      }
      else
      {
        in_word = true;
      }
    }
    if (in_word)
      tokens.push_back(1);
    return tokens;
  }

  /**
   * @brief Given a list of token ID sequences, return their decoded texts
   * @param tokens_list The list of token ID sequences
   * @return The decoded texts as a vector of strings
   */
  std::vector<std::string> decode_batch(const std::vector<std::vector<int>>& tokens_list) const override
  {
    // Simple tokenizer cannot reconstruct exact text from tokens; return token-count strings.
    std::vector<std::string> out;
    out.reserve(tokens_list.size());
    for (const auto& tk : tokens_list)
    {
      out.push_back(std::to_string(static_cast<int>(tk.size())));
    }
    return out;
  }

  /**
   * @brief Given chunked token ID sequences, reconstruct their texts based on original document
   * @param chunk_token The chunked token ID sequences
   * @param doc The original document
   * @param starts The start positions of each chunk in the original token sequence
   * @param lengths The lengths of each chunk
   * @return The decoded chunk texts as a vector of strings
   */
  std::vector<std::string> decode(const std::vector<std::vector<int>>& chunk_token, const std::string& doc,
                                  const std::vector<int>& starts,
                                  const std::vector<int>& lengths) const override
  {
    // Reconstruct chunk texts by slicing the original doc by words according to starts/lengths
    std::vector<std::string> chunk_texts;
    chunk_texts.reserve(chunk_token.size());

    // Split doc into words preserving simple whitespace separation
    std::vector<std::string> words;
    words.reserve(doc.size() / 4);
    std::string current;
    for (char c : doc)
    {
      if (c == ' ' || c == '\n' || c == '\t')
      {
        if (!current.empty())
        {
          words.push_back(current);
          current.clear();
        }
      }
      else
      {
        current.push_back(c);
      }
    }
    if (!current.empty())
      words.push_back(current);

    for (size_t i = 0; i < chunk_token.size(); ++i)
    {
      int start_idx = starts[i];
      int len = lengths[i];
      int end_idx = std::min<int>(start_idx + len, (int)words.size());
      std::string text;
      for (int w = start_idx; w < end_idx; ++w)
      {
        if (!text.empty())
          text.push_back(' ');
        text += words[w];
      }
      chunk_texts.push_back(std::move(text));
    }

    return chunk_texts;
  }

  /**
   * @brief Get the tokenizer type
   * @return The tokenizer type
   */
  TokenizerType type() const override
  {
    return TokenizerType::Simple;
  }
};

}  // namespace nano_graphrag
