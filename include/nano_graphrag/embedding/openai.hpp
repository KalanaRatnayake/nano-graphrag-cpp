#pragma once

#include <string>
#include <vector>
#include <functional>
#include <Poco/JSON/Object.h>
#include <Poco/JSON/Array.h>
#include <Poco/JSON/Parser.h>
#include <Poco/Dynamic/Var.h>
#include "nano_graphrag/interfaces/restapi.hpp"
#include "nano_graphrag/embedding/base.hpp"
#include "nano_graphrag/utils/Log.hpp"

namespace nano_graphrag
{

class OpenAIEmbeddingStrategy : public IEmbeddingStrategy
{
public:
  OpenAIEmbeddingStrategy(size_t dim = 1536, size_t max_tokens = 8192)
    : embedding_dim_(dim), max_token_size_(max_tokens)
  {
  }

  std::vector<std::vector<float>> embed(const std::vector<std::string>& texts) const override
  {
    if (texts.empty())
      return {};
    debug_log("[OpenAIEmbedding] batch=", texts.size());
    using namespace Poco::JSON;
    Object::Ptr body = new Object();
    Array::Ptr arr = new Array();
    for (const auto& t : texts)
      arr->add(t);
    body->set("input", arr);
    body->set("model", "text-embedding-3-small");  // or configurable
    body->set("encoding_format", "float");

    nano_graphrag::RestClient client;
    client.set_uri("https://api.openai.com/v1/embeddings");
    client.set_method("POST");
    client.set_ssl_verify(false);
    const char* key = std::getenv("OPENAI_API_KEY");
    if (!key)
      throw std::runtime_error("OPENAI_API_KEY not set");
    client.set_auth_bearer(key);

    Object::Ptr response = client.post_json(*body, "https://api.openai.com/v1/embeddings");
    std::vector<std::vector<float>> result;
    if (response->has("data"))
    {
      Array::Ptr data = response->getArray("data");
      debug_log("[OpenAIEmbedding] response items=", data->size());
      for (size_t i = 0; i < data->size(); ++i)
      {
        Object::Ptr item = data->getObject(i);
        if (item->has("embedding"))
        {
          Array::Ptr emb = item->getArray("embedding");
          std::vector<float> vec;
          for (size_t j = 0; j < emb->size(); ++j)
          {
            vec.push_back(static_cast<float>(emb->get(j).convert<double>()));
          }
          result.push_back(std::move(vec));
        }
      }
    }
    return result;
  }

  size_t embedding_dim() const override
  {
    return embedding_dim_;
  }
  size_t max_token_size() const override
  {
    return max_token_size_;
  }

private:
  size_t embedding_dim_;
  size_t max_token_size_;
};

}  // namespace nano_graphrag
