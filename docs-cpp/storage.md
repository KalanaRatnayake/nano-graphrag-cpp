# Storage Integration (C++)

This project links the `nano-vectordb-cpp` library to support vector operations. Additional storage backends will be layered on top via strategy interfaces.

## Current Integration

- `nano-vectordb-cpp` is added under `external/` and linked as an interface target:
	- `target_include_directories(nano_vectordb_cpp INTERFACE external/nano-vectordb/include)`
	- `target_link_libraries(nano_vectordb_cpp INTERFACE Eigen3::Eigen OpenSSL::SSL OpenSSL::Crypto nlohmann_json::nlohmann_json)`
	- `nano_graphrag` links `nano_vectordb_cpp` so headers are available across the project.

See: CMakeLists.txt and external/nano-vectordb-cpp/

## Planned Work

- Define C++ storage strategy interfaces mirroring Python (`vdb_*` and `gdb_*`).
- Implement concrete backends:
	- **VectorDB**: HNSW, NanoVectorDB.
	- **GraphDB**: NetworkX-like in-memory graph, Neo4j client.
- Provide factories to select backends similar to tokenizers and chunkers.

## Notes

- Keep storage strategies header-only where practical.
- Align on shared `Types.hpp` for data structures like `TextChunk` to ensure cross-module compatibility.

