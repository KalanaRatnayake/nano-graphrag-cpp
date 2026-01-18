// Abstract base class for embedding strategies
#pragma once
#include <string>
#include <vector>
namespace nano_graphrag
{

/**
 * @brief Abstract base class for embedding strategies
 */
class IEmbeddingStrategy
{
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~IEmbeddingStrategy() = default;

  /**
   * @brief Given a list of texts, return their embeddings
   * @param texts The input texts
   * @return The embeddings as a vector of float vectors
   */
  virtual std::vector<std::vector<float>> embed(const std::vector<std::string>& texts) const = 0;

  /**
   * @brief Optionally: expose embedding dimension and max token size
   * @return The embedding dimension
   */
  virtual size_t embedding_dim() const = 0;

  /**
   * @brief Optionally: expose max token size
   * @return The max token size
   */
  virtual size_t max_token_size() const = 0;
};

}  // namespace nano_graphrag
