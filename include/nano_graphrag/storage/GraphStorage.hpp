#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <tuple>
#include <optional>
#include <algorithm>

#include "nano_graphrag/storage/base.hpp"
#include "nano_graphrag/utils/Types.hpp"

namespace nano_graphrag
{

/**
 * @brief In-memory graph storage backend.
 *
 * Stores nodes and undirected edges with property maps, maintains adjacency,
 * and provides simple clustering via connected components. Intended for
 * lightweight graph operations without external dependencies.
 */
class InMemoryGraphStorage : public BaseGraphStorage
{
public:
  explicit InMemoryGraphStorage(const std::string& ns = "",
                                const std::unordered_map<std::string, std::string>& cfg = {})
  {
    this->namespace_name = ns;
    this->global_config = cfg;
  }

  /** Check if a node exists. */
  bool has_node(const std::string& node_id) const override
  {
    return nodes_.count(node_id) > 0;
  }

  /** Check if an undirected edge exists between nodes s and t. */
  bool has_edge(const std::string& s, const std::string& t) const override
  {
    auto k = canonical_edge_key_str(s, t);
    return edges_.count(k) > 0;
  }

  /** Number of neighbors of the node. */
  int node_degree(const std::string& node_id) const override
  {
    auto it = adjacency_.find(node_id);
    if (it == adjacency_.end())
      return 0;
    return static_cast<int>(it->second.size());
  }

  /** Heuristic edge degree: sum of endpoint degrees. */
  int edge_degree(const std::string& s, const std::string& t) const override
  {
    return node_degree(s) + node_degree(t);
  }

  /** Retrieve node properties, if present. */
  std::optional<std::unordered_map<std::string, std::string>>
  get_node(const std::string& node_id) const override
  {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end())
      return std::nullopt;
    return it->second;
  }

  /** Retrieve edge properties for undirected pair {s,t}, if present. */
  std::optional<std::unordered_map<std::string, std::string>> get_edge(const std::string& s,
                                                                       const std::string& t) const override
  {
    auto k = canonical_edge_key_str(s, t);
    auto it = edges_.find(k);
    if (it == edges_.end())
      return std::nullopt;
    return it->second;
  }

  /** List incident edges (as pairs) for a node. */
  std::vector<std::pair<std::string, std::string>> get_node_edges(const std::string& node_id) const override
  {
    std::vector<std::pair<std::string, std::string>> out;
    auto it = adjacency_.find(node_id);
    if (it == adjacency_.end())
      return out;
    for (const auto& tgt : it->second)
      out.emplace_back(node_id, tgt);
    return out;
  }

  /** Upsert node and its properties. */
  void upsert_node(const std::string& node_id,
                   const std::unordered_map<std::string, std::string>& node_data) override
  {
    nodes_[node_id] = node_data;
    adjacency_.emplace(node_id, std::unordered_set<std::string>{});
  }

  /** Batch upsert nodes. */
  void upsert_nodes_batch(
      const std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>>& nodes_data)
      override
  {
    for (const auto& kv : nodes_data)
      upsert_node(kv.first, kv.second);
  }

  /** Upsert undirected edge and its properties. */
  void upsert_edge(const std::string& s, const std::string& t,
                   const std::unordered_map<std::string, std::string>& edge_data) override
  {
    auto k = canonical_edge_key_str(s, t);
    edges_[k] = edge_data;
    adjacency_[s].insert(t);
    adjacency_[t].insert(s);
  }

  /** Batch upsert edges. */
  void upsert_edges_batch(
      const std::vector<std::tuple<std::string, std::string, std::unordered_map<std::string, std::string>>>&
          edges_data) override
  {
    for (const auto& e : edges_data)
      upsert_edge(std::get<0>(e), std::get<1>(e), std::get<2>(e));
  }

  /**
   * @brief Perform clustering and annotate nodes.
   *
   * Placeholder implementation: clusters by connected components (BFS) and
   * writes a `clusters` attribute on nodes as a simple JSON string.
   */
  void clustering(const std::string& algorithm) override
  {
    // Simple placeholder: mark all nodes level 0 cluster by connected components
    // Assign clusters by BFS
    int cluster_id = 0;
    std::unordered_set<std::string> visited;
    for (const auto& n : nodes_)
    {
      if (visited.count(n.first))
        continue;
      // BFS
      std::vector<std::string> queue{ n.first };
      visited.insert(n.first);
      while (!queue.empty())
      {
        auto cur = queue.back();
        queue.pop_back();
        // store cluster info as JSON-like strings
        nodes_[cur]["clusters"] =
            std::string("[{\"level\":0,\"cluster\":") + std::to_string(cluster_id) + "}]";
        for (const auto& nb : adjacency_[cur])
          if (!visited.count(nb))
          {
            visited.insert(nb);
            queue.push_back(nb);
          }
      }
      cluster_id++;
    }
  }

  /**
   * @brief Construct a simple community schema by grouping nodes by cluster id.
   *
   * Edges are canonicalized and uniqued; `occurrence` is derived from chunk_ids
   * count if present.
   */
  std::unordered_map<std::string, SingleCommunity> community_schema() const override
  {
    std::unordered_map<std::string, SingleCommunity> out;
    // Build communities from node "clusters" attribute
    for (const auto& n : nodes_)
    {
      auto itc = n.second.find("clusters");
      if (itc == n.second.end())
        continue;
      // Extremely simple parse: expect one cluster id
      int cluster = 0;  // default
      try
      {
        auto pos = itc->second.find("cluster\":");
        if (pos != std::string::npos)
        {
          cluster = std::stoi(itc->second.substr(pos + 9));
        }
      }
      catch (...)
      {
      }
      auto key = std::to_string(cluster);
      auto& comm = out[key];
      comm.level = 0;
      comm.title = std::string("Cluster ") + key;
      comm.nodes.push_back(n.first);
      // edges accumulation
      auto eds = get_node_edges(n.first);
      for (auto& e : eds)
      {
        auto a = e.first, b = e.second;
        if (a > b)
          std::swap(a, b);
        comm.edges.emplace_back(a, b);
      }
    }
    // Unique edges per community
    for (auto& kv : out)
    {
      auto& edges = kv.second.edges;
      std::sort(edges.begin(), edges.end());
      edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
      // occurrence heuristic by chunk_ids count
      kv.second.occurrence =
          kv.second.chunk_ids.empty() ? 0.0 : static_cast<double>(kv.second.chunk_ids.size());
    }
    return out;
  }

private:
  static inline std::pair<std::string, std::string> canonical_edge_key(const std::string& s,
                                                                       const std::string& t)
  {
    return s < t ? std::make_pair(s, t) : std::make_pair(t, s);
  }

  static inline std::string canonical_edge_key_str(const std::string& s, const std::string& t)
  {
    auto p = canonical_edge_key(s, t);
    return p.first + "|" + p.second;
  }

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> nodes_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> edges_;
  std::unordered_map<std::string, std::unordered_set<std::string>> adjacency_;
};

}  // namespace nano_graphrag
