#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nano_graphrag/GraphRAG.hpp"
#include "nano_graphrag/embedding/factory.hpp"
#include "nano_graphrag/llm/factory.hpp"
#include "nano_graphrag/utils/Types.hpp"

static std::string read_file(const std::string& path)
{
  std::ifstream ifs(path);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

int main(int argc, char** argv)
{
  using namespace nano_graphrag;

  const char* api_key = std::getenv("OPENAI_API_KEY");
  std::string data_path = "tests/mock_data.txt";  // Dickens A Christmas Carol
  if (argc > 1)
  {
    data_path = argv[1];
  }

  auto start_total = std::chrono::steady_clock::now();

  GraphRAG rag("./nano_cache");

  // Strategies
  std::unique_ptr<IEmbeddingStrategy> emb_up;
  if (api_key)
    emb_up = create_embedding_strategy(EmbeddingStrategyType::OpenAI);
  else
    emb_up = create_embedding_strategy(EmbeddingStrategyType::Hash);
  std::shared_ptr<IEmbeddingStrategy> emb(std::move(emb_up));
  rag.set_embedding_strategy(emb);

  {
    auto llm_up = create_llm_strategy(LLMStrategyType::OpenAI);
    std::shared_ptr<ILLMStrategy> llm(std::move(llm_up));
    rag.set_llm_strategy(llm);
  }

  rag.enable_naive(true);

  // Indexing
  auto start_index = std::chrono::steady_clock::now();
  std::string corpus = read_file(data_path);
  // simple paragraph split on blank lines
  std::vector<std::string> docs;
  {
    std::stringstream ss(corpus);
    std::string line, para;
    while (std::getline(ss, line))
    {
      if (line.empty())
      {
        if (!para.empty())
        {
          docs.push_back(para);
          para.clear();
        }
      }
      else
      {
        if (!para.empty())
          para.push_back('\n');
        para += line;
      }
    }
    if (!para.empty())
      docs.push_back(para);
  }
  rag.insert(docs);
  auto end_index = std::chrono::steady_clock::now();

  // Query
  QueryParam qp;
  qp.mode = "naive";
  qp.top_k = 5;
  qp.response_type = "Multiple Paragraphs";
  qp.naive_max_token_for_text_unit = 4096;
  if (!api_key)
  {
    // Offline mode: measure indexing and retrieve context only
    qp.only_need_context = true;
  }

  std::string question = "What are the top themes in this story?";
  auto start_query = std::chrono::steady_clock::now();
  auto answer = rag.query(question, qp);
  auto end_query = std::chrono::steady_clock::now();

  auto end_total = std::chrono::steady_clock::now();

  auto dur_index = std::chrono::duration_cast<std::chrono::milliseconds>(end_index - start_index).count();
  auto dur_query = std::chrono::duration_cast<std::chrono::milliseconds>(end_query - start_query).count();
  auto dur_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();

  std::cout << "Index time (ms): " << dur_index << "\n";
  std::cout << "Query time (ms): " << dur_query << "\n";
  std::cout << "Total time (ms): " << dur_total << "\n\n";

  std::cout << "Question:\n" << question << "\n\n";
  std::cout << "Answer:\n" << answer << std::endl;

  return 0;
}
