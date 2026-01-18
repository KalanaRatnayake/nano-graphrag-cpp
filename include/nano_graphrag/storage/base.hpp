#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <utility>

// EmbeddingFunc declaration is expected from Embedding.hpp (to be provided)
// Types are defined under utils
#include "nano_graphrag/utils/Types.hpp"
#include "nano_graphrag/embedding/base.hpp"

namespace nano_graphrag
{

/**
 * @brief Common namespace and lifecycle callbacks for storage backends.
 *
 * Provides a logical `namespace_name` to scope data, a `global_config` map for
 * backend-specific settings, and optional lifecycle callbacks invoked during
 * indexing/query phases.
 */
struct StorageNameSpace
{
  std::string namespace_name;
  std::unordered_map<std::string, std::string> global_config;

  inline virtual ~StorageNameSpace() = default;
  /**
   * @brief Callback invoked before a batch index/upsert starts.
   */
  inline virtual void index_start_callback()
  {
  }
  /**
   * @brief Callback invoked after a batch index/upsert completes.
   */
  inline virtual void index_done_callback()
  {
  }
  /**
   * @brief Callback invoked after a query completes.
   */
  inline virtual void query_done_callback()
  {
  }
};

/**
 * @brief Abstract base for vector-search storage backends.
 *
 * Implementations provide vector indexing and top-k query over embedded text.
 * The `embedding_func` is a callable used to convert raw content strings into
 * float vectors. `meta_fields` indicates which keys in the input records should
 * be persisted as metadata to return with query results.
 */
class BaseVectorStorage : public StorageNameSpace
{
public:
  /** Embedding strategy used to convert text to vectors. */
  std::shared_ptr<IEmbeddingStrategy> embedding_strategy;
  std::unordered_map<std::string, bool> meta_fields;  // key existence map

  virtual ~BaseVectorStorage() = default;
  /**
   * @brief Query the storage with a raw text string.
   * @param query The input query text.
   * @param top_k Maximum number of nearest results to return.
   * @return A list of result maps including at least `id` and any requested metadata.
   */
  virtual std::vector<std::unordered_map<std::string, std::string>> query(const std::string& query,
                                                                          int top_k) = 0;
  /**
   * @brief Upsert a batch of records into the storage.
   * @param data Map of record id -> map of fields (must include `content` for embedding).
   */
  virtual void
  upsert(const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& data) = 0;
};

// T is value type stored in KV
/**
 * @brief Abstract base for key-value storage of arbitrary value types.
 *
 * KV storage backends store typed values keyed by string ids. Typical uses
 * include full documents, chunk records, and community reports.
 */
template <typename T>
class BaseKVStorage : public StorageNameSpace
{
public:
  virtual ~BaseKVStorage() = default;
  /**
   * @brief List all keys currently present in the store.
   */
  virtual std::vector<std::string> all_keys() = 0;
  /**
   * @brief Retrieve a single value by id.
   * @return Optional value if present.
   */
  virtual std::optional<T> get_by_id(const std::string& id) = 0;
  /**
   * @brief Batch-retrieve values by ids.
   * @return Vector of optionals corresponding to each requested id.
   */
  virtual std::vector<std::optional<T>> get_by_ids(const std::vector<std::string>& ids) = 0;
  /**
   * @brief Filter for ids that do not exist in the store.
   * @param data List of ids to check.
   * @return List of missing ids.
   */
  virtual std::vector<std::string> filter_keys(const std::vector<std::string>& data) = 0;
  /**
   * @brief Upsert a batch of id->value pairs.
   */
  virtual void upsert(const std::unordered_map<std::string, T>& data) = 0;
  /**
   * @brief Drop all data in the store.
   */
  virtual void drop() = 0;
};

/**
 * @brief Abstract base for graph storage backends.
 *
 * Graph storage supports nodes and undirected edges (stored canonically),
 * property maps per node/edge, and clustering/community reporting APIs.
 */
class BaseGraphStorage : public StorageNameSpace
{
public:
  virtual ~BaseGraphStorage() = default;
  /** Check if a node exists. */
  virtual bool has_node(const std::string& node_id) const = 0;
  /** Check if an undirected edge exists between source and target. */
  virtual bool has_edge(const std::string& source_node_id, const std::string& target_node_id) const = 0;
  /** Degree (number of neighbors) of a node. */
  virtual int node_degree(const std::string& node_id) const = 0;
  /** Sum of degrees of both endpoints (simple heuristic). */
  virtual int edge_degree(const std::string& src_id, const std::string& tgt_id) const = 0;

  /** Retrieve node property map. */
  virtual std::optional<std::unordered_map<std::string, std::string>>
  get_node(const std::string& node_id) const = 0;
  /** Retrieve edge property map (undirected canonical key). */
  virtual std::optional<std::unordered_map<std::string, std::string>>
  get_edge(const std::string& source_node_id, const std::string& target_node_id) const = 0;

  /** List edges incident to a node (as pairs). */
  virtual std::vector<std::pair<std::string, std::string>>
  get_node_edges(const std::string& source_node_id) const = 0;  // may be empty

  /** Upsert a single node and its properties. */
  virtual void upsert_node(const std::string& node_id,
                           const std::unordered_map<std::string, std::string>& node_data) = 0;
  /** Batch upsert nodes. */
  virtual void
  upsert_nodes_batch(const std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>>&
                         nodes_data) = 0;

  /** Upsert a single undirected edge and its properties. */
  virtual void upsert_edge(const std::string& source_node_id, const std::string& target_node_id,
                           const std::unordered_map<std::string, std::string>& edge_data) = 0;
  /** Batch upsert edges. */
  virtual void upsert_edges_batch(
      const std::vector<std::tuple<std::string, std::string, std::unordered_map<std::string, std::string>>>&
          edges_data) = 0;

  /** Execute clustering algorithm and update node/graph attributes accordingly. */
  virtual void clustering(const std::string& algorithm) = 0;
  /** Return a `community_schema` view of clusters/communities. */
  virtual std::unordered_map<std::string, SingleCommunity> community_schema() const = 0;
};

}  // namespace nano_graphrag
