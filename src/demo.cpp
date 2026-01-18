#include <iostream>
#include <vector>
#include <cstdlib>

#include "nano_graphrag/GraphRAG.hpp"
#include "nano_graphrag/embedding/factory.hpp"
#include "nano_graphrag/llm/factory.hpp"
#include "nano_graphrag/utils/Types.hpp"

int main(int argc, char** argv)
{
  using namespace nano_graphrag;
  GraphRAG rag("./nano_cache");

  const char* api_key = std::getenv("OPENAI_API_KEY");

  // Setup strategies
  std::unique_ptr<nano_graphrag::IEmbeddingStrategy> emb_up;
  if (api_key)
    emb_up = nano_graphrag::create_embedding_strategy(nano_graphrag::EmbeddingStrategyType::OpenAI);
  else
    emb_up = nano_graphrag::create_embedding_strategy(nano_graphrag::EmbeddingStrategyType::Hash);
  std::shared_ptr<nano_graphrag::IEmbeddingStrategy> emb(std::move(emb_up));
  rag.set_embedding_strategy(emb);

  // LLM strategy is optional for context-only mode. If API key is present, set it up.
  {
    auto llm_up = nano_graphrag::create_llm_strategy(nano_graphrag::LLMStrategyType::OpenAI);
    std::shared_ptr<nano_graphrag::ILLMStrategy> llm(std::move(llm_up));
    rag.set_llm_strategy(llm);
  }

  rag.enable_naive(true);

  rag.insert({
      "NanoGraphRAG is a lightweight GraphRAG implementation using simple storages.",
      "OpenAI embeddings and chat completions can be used for RAG responses.",
  });

  QueryParam qp;
  qp.mode = "naive";
  qp.response_type = "Multiple Paragraphs";
  qp.top_k = 1;                             // limit to the single most relevant chunk
  qp.naive_max_token_for_text_unit = 1024;  // cap context size
  if (!api_key)
  {
    // Offline demo path: return context only
    qp.only_need_context = true;
  }
  std::string question = "What is NanoGraphRAG?";
  if (argc > 1)
  {
    question = argv[1];
  }
  std::string answer = rag.query(question, qp);
  std::cout << "Question:\n" << question << "\n\n";
  std::cout << "Answer:\n" << answer << std::endl;

  return 0;
}
