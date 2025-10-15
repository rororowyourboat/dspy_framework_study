# DSPy Framework Study

This repository contains small, self-contained scripts that illustrate
how to use [DSPy](https://github.com/stanfordnlp/dspy) for marketing
workflows. Each example assumes you have access to a chat-capable model
(such as ``openai/gpt-4o-mini``) and demonstrates a different facet of
the framework.

## Getting started

1. Create and activate a Python 3.13 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # or `pip install dspy`
   ```
3. Export your model provider credentials so DSPy can authenticate:
   ```bash
   export OPENAI_API_KEY="sk-your-key"
   export DSPY_MODEL="openai/gpt-4o-mini"  # optional override
   ```

> **Tip:** DSPy is provider-agnostic. Replace ``DSPY_MODEL`` with any
> LiteLLM-compatible identifier (Anthropic, Azure OpenAI, local models
> via Ollama, etc.).

## Example scripts

All examples live in the [`examples/`](examples) directory and can be
run individually with ``python``. They are designed to be read as much
as executed—each script includes rich inline documentation and prints
its progress to standard output.

### 1. `basic_showcase.py`

Highlights core DSPy abstractions (signatures, `Predict`,
`ChainOfThought`, and custom modules) by turning a marketing persona into
an email campaign. The script configures an LM from environment
variables, classifies sentiment for a customer quote, walks through a
Chain-of-Thought reasoning example, and finally composes a nurture email
using a bespoke DSPy module.

### 2. `react_campaign_assistant.py`

Demonstrates how DSPy's `ReAct` agent can orchestrate proprietary
toolkits. The agent answers a go-to-market planning question by calling
three small Python utilities that surface competitor intel, estimate ROI
for a campaign budget, and recommend the most effective follow-up
channel.

### 3. `marketing_optimizer.py`

Implements a lightweight optimization loop tailored for growth teams. It
builds a composable DSPy program that reasons about messaging angles and
then generates channel copy. After measuring baseline quality with a
heuristic marketing metric, the script runs `MIPROv2` to compile an
improved version of the program and prints before/after evaluation
scores along with a final copy suggestion.

## Project structure

```
.
├── DSPy_docs.md        # Reference guide for DSPy concepts
├── examples/           # Executable demonstration scripts
├── main.py             # Default entry point (hello world)
├── pyproject.toml      # Project metadata
└── README.md           # This file
```

Feel free to extend the examples or wire them into a larger application
as you explore DSPy's declarative approach to LLM development.
