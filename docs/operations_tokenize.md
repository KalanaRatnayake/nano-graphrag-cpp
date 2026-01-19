# Tokenization (C++)

Tokenizer strategies are used across operations (e.g., chunking). This module provides interchangeable tokenizers.

## Interfaces

- **`ITokenizerStrategy`**:
	- **`encode(text)`**: Returns `std::vector<int>` token IDs.
	- **`decode_batch(tokens_list)`**: Decodes a batch to strings.
	- **`decode(chunk_token, doc, starts, lengths)`**: Decodes chunked token sequences (some strategies ignore `doc/starts/lengths`).
	- **`type()`**: Returns the tokenizer type.

See: include/nano_graphrag/operations/tokenize/base.hpp

## Implementations

- **SimpleTokenizer**:
	- Whitespace-based word tokenization.
	- `decode` reconstructs chunk text by slicing words from the original `doc` using `starts` and `lengths`.

- **TiktokenTokenizer (cpp-tiktoken)**:
	- Uses `cpp-tiktoken` BPE encodings via `GptEncoding`.
	- Default encoding: `LanguageModel::CL100K_BASE`.
	- Supports other encodings like `O200K_BASE`.
	- Accurate `encode` and `decode` based on model files.

See: include/nano_graphrag/operations/tokenize/simple.hpp, openai.hpp, tiktoken.hpp

## Factory

- **`create_tokenizer_strategy(TokenizerType)`**:
	- `TokenizerType::Simple`
	- `TokenizerType::Tiktoken`

See: include/nano_graphrag/operations/tokenize/factory.hpp

## Usage Examples

```cpp
#include "nano_graphrag/operations/tokenize/factory.hpp"
#include "nano_graphrag/operations/tokenize/base.hpp"

auto tok = nano_graphrag::create_tokenizer_strategy(nano_graphrag::TokenizerType::Tiktoken);
auto ids = tok->encode("Hello, world!");
auto txt = tok->decode_batch({ids});

// If you need a specific encoding:
#include "encoding.h"
auto custom = std::make_unique<nano_graphrag::TiktokenTokenizer>(LanguageModel::O200K_BASE);
```

## Build Integration

- The `cpp-tiktoken` library is included via `add_subdirectory(external/cpp-tiktoken)` and linked to the interface library.
- It brings a `pcre2` dependency which is built automatically.

## Notes

- Choose `TiktokenTokenizer` for accurate chunking and limits.
- `SimpleTokenizer` is useful for smoke tests or environments without external dependencies.

