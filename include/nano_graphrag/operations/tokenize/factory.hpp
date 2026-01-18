#pragma once
#include <memory>
#include "nano_graphrag/operations/tokenize/base.hpp"
#include "nano_graphrag/operations/tokenize/simple.hpp"
#include "nano_graphrag/operations/tokenize/tiktoken.hpp"

namespace nano_graphrag
{

inline std::unique_ptr<ITokenizerStrategy> create_tokenizer_strategy(TokenizerType type)
{
  switch (type)
  {
    case TokenizerType::Simple:
      return std::make_unique<SimpleTokenizer>();
    case TokenizerType::Tiktoken:
      return std::make_unique<TiktokenTokenizer>();
    default:
      return nullptr;
  }
}

}  // namespace nano_graphrag
