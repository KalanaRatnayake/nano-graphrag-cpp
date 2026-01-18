#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include "nano_graphrag/embedding/base.hpp"

namespace nano_graphrag {

class HashEmbeddingStrategy : public IEmbeddingStrategy {
public:
  explicit HashEmbeddingStrategy(size_t dim = 256, size_t max_tokens = 8192)
    : dim_(dim), max_tokens_(max_tokens) {}

  std::vector<std::vector<float>> embed(const std::vector<std::string>& texts) const override {
    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto& t : texts) {
      std::vector<float> vec(dim_, 0.0f);
      std::istringstream iss(t);
      std::string tok;
      size_t count = 0;
      while (iss >> tok) {
        auto h = std::hash<std::string>{}(tok);
        size_t idx = h % dim_;
        vec[idx] += 1.0f;
        if (++count >= max_tokens_) break;
      }
      // L2 normalize
      float norm = 0.0f;
      for (float v : vec) norm += v * v;
      norm = std::sqrt(norm);
      if (norm > 0.0f) {
        for (float& v : vec) v /= norm;
      }
      out.push_back(std::move(vec));
    }
    return out;
  }

  size_t embedding_dim() const override { return dim_; }
  size_t max_token_size() const override { return max_tokens_; }

private:
  size_t dim_;
  size_t max_tokens_;
};

} // namespace nano_graphrag
