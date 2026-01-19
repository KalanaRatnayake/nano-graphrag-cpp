# Nano-graphrag-cpp

`nano-graphrag-cpp` is a minimal C++ implementation of [GraphRAG](https://arxiv.org/pdf/2404.16130), inspired by the Python [nano-graphrag](https://github.com/gusye1234/nano-graphrag) project. It focuses on:

- Strategy-based modularity for embeddings, LLMs, tokenization, chunking, and storage
- Fast, in-memory and persistent vector search (nano-vectordb-cpp backend)
- Pluggable graph storage (in-memory only currently, extendable)
- Simple, extensible APIs for RAG, entity extraction, and graph-based retrieval
- Out-of-the-box support for OpenAI embeddings

Core concepts:
- GraphRAG: Orchestrates chunking, embedding, storage, and retrieval for RAG
- Strategies: Embedding, LLM, tokenizer, chunking, and storage are all pluggable
- Storage: Uses nano-vectordb-cpp for vector search, with optional SQLite/file persistence
- Modes: Supports naive, local, and global retrieval modes (see test_christmas_carol.cpp)

Typical flow:
1. Construct GraphRAG with a working directory
2. Set embedding and LLM strategies (OpenAI etc.)
3. Insert documents (auto-chunked and embedded)
4. Query in naive, local, or global mode
5. Reload from disk for persistent workflows

> **Note:** This branch is pure C++ implementation and currently only limited interfaces only. The Python reference for feature parity are tracked in `reference-python` branch. See the original [Python version](https://github.com/gusye1234/nano-graphrag) for more advanced graph/entity features.

## Requirements

- C++17 or later
- [Eigen](https://eigen.tuxfamily.org/) (vector math)
- [OpenSSL](https://www.openssl.org/) (base64 encoding)
- [nlohmann/json](https://github.com/nlohmann/json) (JSON serialization)
- [Poco](https://pocoproject.org/) (HTTP/JSON/Net)
- [SQLite3](https://www.sqlite.org/)

### Build & Test (CMake)

```bash
sudo apt-get update
sudo apt-get install -y g++ cmake libeigen3-dev libssl-dev nlohmann-json3-dev libpoco-dev libsqlite3-dev
```

### Build using CMake

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Run the demo and benchmark

Please set OpenAI API key in environment: `export OPENAI_API_KEY="sk-..."`

Then run

```bash
./demo
```
or 

```bash
./test_christmas_carol
```

Christmas carol querying results are compared with results from the original and other are [available here](./docs/benchmark-en.md).

>Note that the results are not performance/timing comparisons but only functionality comparisons.

## Usage

```cpp
#include "nano_graphrag/GraphRAG.hpp"
using namespace nano_graphrag;

GraphRAG rag("./cache");
rag.set_embedding_strategy(create_embedding_strategy(EmbeddingStrategyType::OpenAI));
rag.set_llm_strategy(create_llm_strategy(LLMStrategyType::OpenAI));
rag.enable_naive(true);
rag.insert({"A Christmas Carol by Charles Dickens ..."});
QueryParam qp;
qp.mode = "global";
std::string answer = rag.query("What are the top themes in this story?", qp);
```

See `src/demo.cpp` and `src/test_christmas_carol.cpp` for full API usage and tests.

## Documentation

- [Embedding strategies](./docs/embedding.md)
- [LLM strategies](./docs/llm.md)
- [Chunking operations](./docs/operations_chunking.md)
- [Tokenization operations](./docs/operations_tokenize.md)
- [Storage backends](./docs/storage.md)
