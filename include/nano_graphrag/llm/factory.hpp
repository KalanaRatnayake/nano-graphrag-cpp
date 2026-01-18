#pragma once
#include "nano_graphrag/llm/base.hpp"
#include "nano_graphrag/llm/openai.hpp"
#include <memory>

namespace nano_graphrag
{

/**
 * @brief Enum for different LLM strategy types
 *
 * @param OpenAI OpenAI LLM strategy
 */
enum class LLMStrategyType
{
  OpenAI,
  // Add more strategies here
};

/**
 * @brief Factory function to create LLM strategy instances
 *
 * @param type The type of LLM strategy to create
 * @return std::unique_ptr<ILLMStrategy> The created LLM strategy instance
 */
inline std::unique_ptr<ILLMStrategy> create_llm_strategy(LLMStrategyType type)
{
  switch (type)
  {
    case LLMStrategyType::OpenAI:
      return std::make_unique<OpenAILLMStrategy>();
    // Add more cases here
    default:
      return nullptr;
  }
}

}  // namespace nano_graphrag
