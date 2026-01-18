#pragma once
#include <memory>
#include "nano_graphrag/embedding/base.hpp"
#include "nano_graphrag/embedding/openai.hpp"
#include "nano_graphrag/embedding/hash.hpp"

namespace nano_graphrag
{

/**
 * @brief Enum for different embedding strategy types
 *
 * @param OpenAI OpenAI embedding strategy
 */
enum class EmbeddingStrategyType
{
  OpenAI,
  Hash,
  // Add more strategies here
};

/**
 * @brief Factory function to create embedding strategy instances
 *
 * @param type The type of embedding strategy to create
 * @return std::unique_ptr<IEmbeddingStrategy> The created embedding strategy instance
 */
inline std::unique_ptr<IEmbeddingStrategy> create_embedding_strategy(EmbeddingStrategyType type)
{
  switch (type)
  {
    case EmbeddingStrategyType::OpenAI:
      return std::make_unique<OpenAIEmbeddingStrategy>();
    case EmbeddingStrategyType::Hash:
      return std::make_unique<HashEmbeddingStrategy>();
    // Add more cases here
    default:
      return nullptr;
  }
}

}  // namespace nano_graphrag
