#pragma once
#include <memory>
#include "nano_graphrag/operations/chunking/base.hpp"
#include "nano_graphrag/operations/chunking/default.hpp"

namespace nano_graphrag
{

/**
 * @brief Enum for different chunking strategy types
 */
enum class ChunkingStrategyType
{
  Default,
  // Add more strategies here when available
};

/**
 * @brief Factory function to create chunking strategy instances
 *
 * @param type The type of chunking strategy to create
 * @return std::unique_ptr<IChunkingStrategy> The created chunking strategy instance
 */
inline std::unique_ptr<IChunkingStrategy> create_chunking_strategy(ChunkingStrategyType type)
{
  switch (type)
  {
    case ChunkingStrategyType::Default:
      return std::make_unique<DefaultChunkingStrategy>();
    default:
      return nullptr;
  }
}

}  // namespace nano_graphrag
