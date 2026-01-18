// Abstract base class for LLM strategies
#pragma once
#include <string>
#include <vector>
namespace nano_graphrag
{

/**
 * @brief Abstract base class for LLM strategies
 */
class ILLMStrategy
{
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~ILLMStrategy() = default;

  /**
   * @brief Given a user prompt (and optional system prompt), return the LLM completion
   * @param user_prompt The user prompt
   * @param system_prompt The optional system prompt
   * @return The LLM completion
   */
  virtual std::string prompt(const std::string& user_prompt, const std::string& system_prompt = "") const = 0;

  /**
   * @brief Optionally: expose model name
   * @return The model name
   */
  virtual std::string model_name() const = 0;
};

}  // namespace nano_graphrag
