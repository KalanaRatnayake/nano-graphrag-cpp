# LLM Strategies (C++)

This module provides a strategy-based interface for Large Language Models (LLMs) with a concrete OpenAI implementation.

## Interfaces

- **`ILLMStrategy`**: Abstract interface with:
	- **`prompt(user_prompt, system_prompt)`**: Returns a completion string.
	- **`model_name()`**: Returns the strategy’s model identifier.

See: include/nano_graphrag/llm/base.hpp

## Implementations

- **OpenAILLMStrategy**:
	- Calls OpenAI Responses API: `POST /v1/responses` using `Poco::JSON` and `RestClient`.
	- Payload uses `model`, `input` (user prompt), and optional `instructions` (system prompt).
	- Default model: `gpt-3.5-turbo` (pass a name in the constructor to override).
	- Requires `OPENAI_API_KEY`.
	- Parses `output_text` when available; falls back to structured `output[].content[].text`, and finally legacy `choices[0].message.content`.

See: include/nano_graphrag/llm/openai.hpp

## Factory

- **`create_llm_strategy(LLMStrategyType)`** returns a concrete strategy.
	- Supported: `LLMStrategyType::OpenAI`.

See: include/nano_graphrag/llm/factory.hpp

## Usage Example

```cpp
#include "nano_graphrag/llm/factory.hpp"
#include <iostream>

int main() {
	auto llm = nano_graphrag::create_llm_strategy(nano_graphrag::LLMStrategyType::OpenAI);
	std::string reply = llm->prompt("Summarize GraphRAG in one sentence.");
	std::cout << reply << std::endl;
}
```

## Environment

```bash
export OPENAI_API_KEY=sk-...
```

## Notes

- Prefers the Responses API’s `output_text` convenience field.
- For structured outputs or JSON, extend the strategy to set `response_format` or parse `output` content items.

