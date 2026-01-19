# Embedding Strategies (C++)

This module provides a strategy-based interface for text embeddings. It mirrors the Python design but is implemented in C++ with concrete integrations.

## Interfaces

- **`IEmbeddingStrategy`**: Abstract interface with:
	- **`embed(texts)`**: Returns `std::vector<std::vector<float>>` for a batch of input strings.
	- **`embedding_dim()`**: Embedding vector dimension.
	- **`max_token_size()`**: Maximum token length supported by the strategy.

See: include/nano_graphrag/embedding/base.hpp

## Implementations

- **OpenAIEmbeddingStrategy**:
	- Calls OpenAI `POST /v1/embeddings` using `Poco::JSON` and the shared `RestClient`.
	- Default model: `text-embedding-3-small` (configurable at construction).
	- Requires environment variable `OPENAI_API_KEY`.
	- Produces `float` vectors from the JSON response’s `data[*].embedding`.

See: include/nano_graphrag/embedding/openai.hpp

## Factory

- **`create_embedding_strategy(EmbeddingStrategyType)`** returns a concrete strategy.
	- Supported: `EmbeddingStrategyType::OpenAI`.

See: include/nano_graphrag/embedding/factory.hpp

## Usage Example

```cpp
#include "nano_graphrag/embedding/factory.hpp"
#include <iostream>

int main() {
	auto emb = nano_graphrag::create_embedding_strategy(nano_graphrag::EmbeddingStrategyType::OpenAI);
	std::vector<std::string> inputs = {"hello world", "graph rag"};
	auto vectors = emb->embed(inputs);
	std::cout << "dim=" << emb->embedding_dim() << ", count=" << vectors.size() << std::endl;
}
```

## Environment

- Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

## Extending

- Add new strategies by implementing `IEmbeddingStrategy` and wiring them in the factory.
- For non-HTTP models, ensure consistent batch behavior and stable `embedding_dim()`.

