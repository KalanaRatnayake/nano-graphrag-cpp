# Chunking (C++)

Chunking transforms large documents into manageable windows for downstream processing.

## Interfaces

- **`IChunkingStrategy`**:
	- **`chunk(doc)`**: Returns `std::vector<std::string>` chunks for a single document.
	- **`set_tokenizer(shared_ptr<ITokenizerStrategy>)`**: Injects the tokenizer used to measure token sizes and decode chunk text.
	- **`set_chunk_size(size)`**: Sets the max tokens per chunk.
	- **`set_overlap_size(size)`**: Sets the overlap between consecutive chunks.

See: include/nano_graphrag/operations/chunking/base.hpp

## Default Strategy

- **DefaultChunkingStrategy**:
	- Token-based sliding window with overlap: windows advance by `max_token_size - overlap_token_size`.
	- Delegates text reconstruction to the injected tokenizer’s `decode(...)`.
	- Defaults: `chunk_size = 1024`, `overlap_size = 128`.
	- Helper: `chunking_by_token_size(tokens_list, docs, doc_keys, overlap, max)` produces `TextChunk` records, used by `get_chunks(...)`.

See: include/nano_graphrag/operations/chunking/default.hpp and utils types in include/nano_graphrag/utils/Types.hpp

## Usage Example

```cpp
#include "nano_graphrag/operations/chunking/factory.hpp"
#include "nano_graphrag/operations/tokenize/factory.hpp"

auto chunker = nano_graphrag::create_chunking_strategy(nano_graphrag::ChunkingStrategyType::Default);
auto tokenizer = nano_graphrag::create_tokenizer_strategy(nano_graphrag::TokenizerType::Tiktoken);
chunker->set_tokenizer(std::move(tokenizer));
chunker->set_chunk_size(1024);
chunker->set_overlap_size(128);

std::string doc = "GraphRAG enables structured retrieval over knowledge graphs...";
auto chunks = chunker->chunk(doc);
```

## Notes

- Use `TiktokenTokenizer` for accurate limits aligned to your LLM.
- Overlap helps context continuity but increases total tokens; tune based on downstream model limits.

