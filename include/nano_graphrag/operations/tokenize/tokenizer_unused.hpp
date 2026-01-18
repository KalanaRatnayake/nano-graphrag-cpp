#pragma once

#include <string>
#include <vector>

namespace nano_graphrag
{

class TokenizerWrapper
{
public:
  enum class Type
  {
    Simple,
    Tiktoken,
    HuggingFace
  };

  explicit TokenizerWrapper(Type t = Type::Simple, const std::string& model = "")
    : type_(t), model_name_(model)
  {
  }

  // For Simple type: treat each whitespace-separated word as a token
  std::vector<int> encode(const std::string& text) const
  {
    if (type_ != Type::Simple)
    {
      // Placeholder for advanced tokenizers; implement when available
      // Fallback to simple behavior for now
    }
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

  // For non-simple types, this would decode token IDs to strings; we stub it.
  std::vector<std::string> decode_batch(const std::vector<std::vector<int>>& tokens_list) const
  {
    std::vector<std::string> out;
    out.reserve(tokens_list.size());
    for (const auto& tk : tokens_list)
    {
      // Placeholder: join count of tokens as a simple string
      out.push_back(std::to_string(static_cast<int>(tk.size())));
    }
    return out;
  }

  Type type() const
  {
    return type_;
  }

private:
  Type type_;
  std::string model_name_;
};

}  // namespace nano_graphrag
