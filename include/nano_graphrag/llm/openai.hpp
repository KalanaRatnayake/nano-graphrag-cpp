#pragma once

#include <string>
#include <stdexcept>
#include <cstdlib>
#include <Poco/JSON/Object.h>
#include <Poco/JSON/Array.h>
#include <Poco/JSON/Parser.h>
#include <Poco/Dynamic/Var.h>
#include "nano_graphrag/interfaces/restapi.hpp"
#include "nano_graphrag/llm/base.hpp"
#include "nano_graphrag/utils/Log.hpp"

namespace nano_graphrag
{

class OpenAILLMStrategy : public ILLMStrategy
{
public:
  OpenAILLMStrategy(const std::string& model = "gpt-3.5-turbo") : model_name_(model)
  {
  }

  std::string prompt(const std::string& user_prompt, const std::string& system_prompt = "") const override
  {
    // Use OpenAI Responses API: POST /v1/responses
    Poco::JSON::Object::Ptr body = new Poco::JSON::Object();
    body->set("model", model_name_);
    body->set("input", user_prompt);
    if (!system_prompt.empty())
    {
      // Provide system instructions when available
      body->set("instructions", system_prompt);
    }

    RestClient client;
    client.set_uri("https://api.openai.com/v1/responses");
    client.set_method("POST");
    client.set_ssl_verify(false);
    const char* key = std::getenv("OPENAI_API_KEY");
    if (!key)
      throw std::runtime_error("OPENAI_API_KEY not set");
    client.set_auth_bearer(key);

    debug_log("[OpenAILLM] sending prompt model=", model_name_);
    Poco::JSON::Object::Ptr response = client.post_json(*body, "https://api.openai.com/v1/responses");
    debug_log("[OpenAILLM] response received");
    // Prefer output_text convenience field
    if (response->has("output_text"))
    {
      return response->getValue<std::string>("output_text");
    }
    // Fallback: parse structured output array
    if (response->has("output"))
    {
      Poco::JSON::Array::Ptr output = response->getArray("output");
      if (!output->empty())
      {
        Poco::JSON::Object::Ptr item = output->getObject(0);
        if (item->has("content"))
        {
          Poco::JSON::Array::Ptr content = item->getArray("content");
          for (size_t i = 0; i < content->size(); ++i)
          {
            Poco::JSON::Object::Ptr c = content->getObject(i);
            if (c->has("text"))
            {
              return c->getValue<std::string>("text");
            }
          }
        }
      }
    }
    // Legacy fallback: in case the server responds with chat completion schema
    if (response->has("choices"))
    {
      Poco::JSON::Array::Ptr choices = response->getArray("choices");
      if (!choices->empty())
      {
        Poco::JSON::Object::Ptr choice = choices->getObject(0);
        if (choice->has("message"))
        {
          Poco::JSON::Object::Ptr msg = choice->getObject("message");
          if (msg->has("content"))
            return msg->getValue<std::string>("content");
        }
      }
    }
    return "";
  }

  std::string model_name() const override
  {
    return model_name_;
  }

private:
  std::string model_name_;
};

}  // namespace nano_graphrag
