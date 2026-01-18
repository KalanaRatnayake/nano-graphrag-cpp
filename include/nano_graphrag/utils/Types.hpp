#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace nano_graphrag {

struct TextChunk {
  int tokens{0};
  std::string content;
  std::string full_doc_id;
  int chunk_order_index{0};
};

struct SingleCommunity {
  int level{0};
  std::string title;
  std::vector<std::pair<std::string, std::string>> edges; // undirected stored as sorted pairs
  std::vector<std::string> nodes;
  std::vector<std::string> chunk_ids;
  double occurrence{0.0};
  std::vector<std::string> sub_communities; // keys to child communities
};

struct Community : public SingleCommunity {
  std::string report_string;
  std::unordered_map<std::string, std::string> report_json; // minimal JSON map
};

struct QueryParam {
  std::string mode{"global"}; // "local" | "global" | "naive"
  bool only_need_context{false};
  std::string response_type{"Multiple Paragraphs"};
  int level{2};
  int top_k{20};

  // naive search
  int naive_max_token_for_text_unit{12000};

  // local search
  int local_max_token_for_text_unit{4000};
  int local_max_token_for_local_context{4800};
  int local_max_token_for_community_report{3200};
  bool local_community_single_one{false};

  // global search
  double global_min_community_rating{0.0};
  int global_max_consider_community{512};
  int global_max_token_for_community_report{16384};

  // extra llm kwargs used for community mapping (JSON response)
  std::unordered_map<std::string, std::string> global_special_community_map_llm_kwargs{
    {"response_format", "json_object"}
  };
};

} // namespace nano_graphrag
