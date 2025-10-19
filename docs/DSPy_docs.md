# DSPy: Declarative Self-improving Python

DSPy is a declarative framework for building modular AI software that enables developers to iterate on structured code rather than brittle prompt strings. It shifts the paradigm from prompt engineering to programming by allowing developers to define AI system behavior through signatures (input/output specifications) and modules, while offering algorithms that automatically compile these programs into effective prompts and weights for language models. This approach makes AI software more reliable, maintainable, and portable across different models and optimization strategies.

The framework is designed for building production-ready AI applications ranging from simple classifiers to sophisticated RAG pipelines and agent loops. Instead of manually crafting prompts or managing training jobs, DSPy enables developers to build AI software from natural-language modules that can be composed generically with different models, inference strategies, and learning algorithms. The core philosophy is analogous to the shift from assembly to C or pointer arithmetic to SQL—providing a higher-level language for AI programming that decouples system design from low-level implementation details.

## Language Model Configuration

Configure your LM once and use it across all modules.

```python
import dspy

# OpenAI
lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_API_KEY")
dspy.configure(lm=lm)

# Anthropic
lm = dspy.LM("anthropic/claude-3-opus-20240229", api_key="YOUR_ANTHROPIC_API_KEY")
dspy.configure(lm=lm)

# Local models via Ollama
lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

# Direct LM calls
lm("Say this is a test!")  # => ['This is a test!']
lm(messages=[{"role": "user", "content": "Hello!"}])  # => ['Hello! How can I help?']
```

## Signatures - Define Input/Output Behavior

Specify what your AI module should do, not how.

```python
# Simple inline signatures
"question -> answer"
"question -> answer: float"
"context, question -> response"

# Typed class signatures with descriptions
from typing import Literal

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

# Complex structured outputs
class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""
    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="list of entities with metadata")

module = dspy.Predict(ExtractInfo)
result = module(text="Apple Inc. announced iPhone 14 today. CEO Tim Cook highlighted new features.")
print(result.title, result.headings, result.entities)
```

## Basic Predictor Module

Map inputs to outputs without reasoning steps.

```python
classify = dspy.Predict("sentence -> sentiment: bool")
response = classify(sentence="It's a charming and often affecting journey.")
print(response.sentiment)  # True
```

## Chain of Thought Reasoning

Add step-by-step reasoning before producing output.

```python
math = dspy.ChainOfThought("question -> answer: float")
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(result.reasoning)
# => 'When two dice are tossed, each die has 6 faces, resulting in 36 possible outcomes...'
print(result.answer)  # => 0.0277776
```

## ReAct Agent with Tools

Build tool-using agents that reason about actions.

```python
def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for information."""
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]

def evaluate_math(expression: str):
    """Evaluate a mathematical expression."""
    return dspy.PythonInterpreter({}).execute(expression)

react = dspy.ReAct(
    "question -> answer: float",
    tools=[evaluate_math, search_wikipedia],
    max_iters=10
)

pred = react(question="What is 9362158 divided by the birth year of David Gregory of Kinnairdy?")
print(pred.answer)  # => 5761.328
print(pred.trajectory)  # Shows all reasoning steps and tool calls
```

## Program of Thought

Generate and execute code to solve problems.

```python
pot = dspy.ProgramOfThought("question -> answer")
result = pot(question="Sarah has 5 apples. She buys 7 more. How many apples does Sarah have?")
print(result.answer)  # => 12
```

## Building Custom RAG Systems

Compose modules into sophisticated pipelines.

```python
class RAG(dspy.Module):
    def __init__(self, num_docs=5):
        super().__init__()
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        # Retrieve relevant documents
        retriever = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
        context = retriever(question, k=self.num_docs)
        context = [doc["text"] for doc in context]

        # Generate response with context
        return self.respond(context=context, question=question)

# Use the RAG system
rag = RAG(num_docs=3)
response = rag(question="What's the name of the castle that David Gregory inherited?")
print(response.response)  # => "Kinnairdy Castle"
```

## Multi-Hop Reasoning Pipeline

Chain multiple reasoning and retrieval steps.

```python
class MultiHop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        super().__init__()
        self.num_docs = num_docs
        self.num_hops = num_hops
        self.generate_query = dspy.ChainOfThought("claim, notes -> query")
        self.append_notes = dspy.ChainOfThought("claim, notes, context -> new_notes: list[str]")
        self.retriever = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")

    def forward(self, claim: str):
        notes = []

        for hop in range(self.num_hops):
            # Generate search query based on claim and accumulated notes
            query = self.generate_query(claim=claim, notes=notes).query

            # Retrieve documents
            context = self.retriever(query, k=self.num_docs)
            context = [doc["text"] for doc in context]

            # Extract new information
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)

        return dspy.Prediction(notes=notes)

multi_hop = MultiHop(num_docs=5, num_hops=3)
result = multi_hop(claim="The 2020 Summer Olympics were postponed to 2021.")
print(result.notes)
```

## Evaluation with Custom Metrics

Measure and optimize your AI systems.

```python
from dspy.evaluate import Evaluate

# Simple boolean metric
def exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Numeric score metric
def parse_integer_answer(answer):
    try:
        answer = answer.strip().split('\n')[0]
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = ''.join([c for c in answer if c.isdigit()])
        return int(answer)
    except (ValueError, IndexError):
        return 0

def gsm8k_metric(gold, pred, trace=None):
    return parse_integer_answer(str(gold.answer)) == parse_integer_answer(str(pred.answer))

# Run evaluation
evaluator = Evaluate(
    devset=devset,
    metric=exact_match,
    num_threads=16,
    display_progress=True,
    display_table=5
)

score = evaluator(your_program)
print(f"Accuracy: {score}")
```

## LLM-as-Judge Metrics

Use LLMs to evaluate complex outputs.

```python
class FactJudge(dspy.Signature):
    """Judge if the answer is factually correct based on the context."""
    context = dspy.InputField(desc="Context for the prediction")
    question = dspy.InputField(desc="Question to be answered")
    answer = dspy.InputField(desc="Answer for the question")
    factually_correct: bool = dspy.OutputField(desc="Is answer factually correct?")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example, pred):
    result = judge(context=example.context, question=example.question, answer=pred.answer)
    return result.factually_correct

evaluator = Evaluate(devset=devset, metric=factuality_metric, num_threads=8)
score = evaluator(rag_program)
```

## MIPROv2 Optimizer - Automatic Prompt Optimization

Optimize instructions and few-shot examples jointly.

```python
from dspy.teleprompt import MIPROv2
from dspy.datasets import HotPotQA

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Define your metric
def answer_exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Load dataset
trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]

# Create unoptimized program
program = dspy.ChainOfThought("question -> answer")

# Optimize with MIPROv2 (auto mode)
optimizer = MIPROv2(
    metric=answer_exact_match,
    auto="light",  # "light", "medium", or "heavy"
    num_threads=24
)

optimized_program = optimizer.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4
)

# Save for production
optimized_program.save("optimized_program.json")

# Evaluate
evaluator = Evaluate(devset=devset, metric=answer_exact_match)
print(f"Before: {evaluator(program)}")
print(f"After: {evaluator(optimized_program)}")
```

## BootstrapFewShot Optimizer

Generate few-shot examples automatically from training data.

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

optimizer = BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    max_errors=10
)

optimized = optimizer.compile(student=program, trainset=trainset)
optimized.save("bootstrapped.json")
```

## COPRO Optimizer - Instruction Optimization

Optimize natural language instructions via coordinate ascent.

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    prompt_model=dspy.LM("openai/gpt-4o"),  # Use strong model for proposals
    metric=your_metric,
    breadth=10,  # Number of instruction candidates per round
    depth=3,     # Number of optimization rounds
    init_temperature=1.0
)

optimized = optimizer.compile(
    your_program,
    trainset=trainset,
    eval_kwargs=dict(num_threads=16, display_progress=True)
)
```

## BootstrapFinetune - Fine-tuning Optimizer

Generate training data and fine-tune model weights.

```python
from dspy.teleprompt import BootstrapFinetune
from typing import Literal

# Define your program
CLASSES = ["transfer", "transactions", "balance", "freeze_account", "activate_card"]
signature = dspy.Signature("text -> label").with_updated_fields("label", type_=Literal[tuple(CLASSES)])
classify = dspy.ChainOfThought(signature)

# Optimize via fine-tuning
optimizer = BootstrapFinetune(
    metric=lambda x, y, trace=None: x.label == y.label,
    num_threads=24
)

config = dict(
    target="openai/gpt-4o-mini-2024-07-18",
    epochs=2,
    bf16=True,
    bsize=6,
    accumsteps=2,
    lr=5e-5
)

optimized = optimizer.compile(classify, trainset=trainset, **config)
optimized(text="What does a pending cash withdrawal mean?")
# => Prediction(reasoning='...', label='pending_cash_withdrawal')
```

## KNN Few-Shot Selection

Dynamically select examples based on similarity.

```python
from dspy.teleprompt import KNNFewShot
from sentence_transformers import SentenceTransformer
from dspy import Embedder

optimizer = KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

optimized = optimizer.compile(student=dspy.ChainOfThought("question -> answer"))
```

## Ensemble Multiple Programs

Combine predictions from multiple programs.

```python
from dspy.teleprompt import Ensemble

# Create multiple candidate programs
programs = [program1, program2, program3]

# Create ensemble with majority voting
ensemble = Ensemble(reduce_fn=dspy.majority)
ensemble_program = ensemble.compile(programs)

# Use ensemble
result = ensemble_program(question="What is DSPy?")
```

## Best-of-N Sampling

Sample multiple times and return the best output.

```python
qa = dspy.ChainOfThought("question -> answer")

def one_word_answer(args, pred):
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=qa,
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # => "Brussels"
```

## Iterative Refinement

Refine outputs iteratively with feedback.

```python
qa = dspy.ChainOfThought("question -> answer")

def concise_answer(args, pred):
    return 1.0 if len(pred.answer) <= 50 else 0.0

refine = dspy.Refine(
    module=qa,
    N=3,
    reward_fn=concise_answer,
    threshold=1.0,
    fail_count=2  # Max failures before raising error
)

result = refine(question="Explain quantum computing")
print(result.answer)
```

## Parallel Module Execution

Execute multiple modules concurrently.

```python
parallel = dspy.Parallel(num_threads=4)
predict = dspy.Predict("question -> answer")

results = parallel([
    (predict, dspy.Example(question="What is 2+2?").with_inputs("question")),
    (predict, dspy.Example(question="What is 3+3?").with_inputs("question")),
    (predict, dspy.Example(question="What is 4+4?").with_inputs("question")),
    (predict, dspy.Example(question="What is 5+5?").with_inputs("question"))
])

for result in results:
    print(result.answer)
```

## Batch Processing

Process multiple examples efficiently.

```python
program = dspy.ChainOfThought("question -> answer")

examples = [
    dspy.Example(question="What is 2+2?").with_inputs("question"),
    dspy.Example(question="What is 3+3?").with_inputs("question"),
    dspy.Example(question="What is 4+4?").with_inputs("question")
]

results = program.batch(
    examples,
    num_threads=8,
    max_errors=2,
    return_failed_examples=True,
    display_progress_bar=True
)

# With error handling
success, failures, exceptions = results
```

## Async Operations

Convert any module to async for concurrent execution.

```python
import asyncio

program = dspy.ChainOfThought("question -> answer")
async_program = dspy.asyncify(program)

async def main():
    result = await async_program(question="What is DSPy?")
    print(result.answer)

asyncio.run(main())
```

## Streaming Responses

Stream outputs token by token.

```python
import asyncio

predict = dspy.Predict("question -> answer")
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")]
)

async def main():
    output_stream = stream_predict(question="Why did the chicken cross the road?")
    async for chunk in output_stream:
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## ColBERT Retrieval

Use ColBERT for high-quality document retrieval.

```python
colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# Simple retrieval
results = colbert(query="What is DSPy?", k=5)
for result in results:
    print(result['text'])

# Use in a module
retriever = dspy.Retrieve(k=3)
dspy.settings.configure(rm=colbert)

topK_passages = retriever(query="When was the first FIFA World Cup held?").passages
for idx, passage in enumerate(topK_passages):
    print(f"{idx+1}] {passage}")
```

## Data Loading from HuggingFace

Load and prepare datasets easily.

```python
from dspy.datasets import DataLoader

loader = DataLoader()

# Load dataset
dataset = loader.from_huggingface(
    dataset_name="PolyAI/banking77",
    split="train",
    fields=("text", "label"),
    input_keys=("text",),
    trust_remote_code=True
)

# Create examples with hints
CLASSES = ["transfer", "balance", "card_issues"]
trainset = [
    dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
    for x in dataset[:1000]
]
```

## Caching Configuration

Control caching behavior for reproducibility or freshness.

```python
# Disable caching entirely
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False
)

# Force fresh outputs with unique rollout_id
predict = dspy.Predict("question -> answer")
predict(question="1+1", config={"rollout_id": 1, "temperature": 1.0})
predict(question="1+1", config={"rollout_id": 2, "temperature": 1.0})  # Fresh call
```

## Usage Tracking

Monitor token consumption and costs.

```python
dspy.settings.configure(track_usage=True)

result = dspy.ChainOfThought("question -> answer")(question="What is 2+2?")
usage = result.get_lm_usage()

print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
```

## Inspect LM Call History

Debug and understand LM interactions.

```python
program = dspy.ChainOfThought("question -> answer")
result = program(question="What is DSPy?")

# Inspect last N calls
dspy.inspect_history(n=1)
# Shows: prompts, responses, tokens used, etc.

# Access module's history
program.inspect_history(n=2)
```

## Save and Load Programs

Persist optimized programs for production deployment.

```python
# Save state only (JSON - recommended)
program.save("program.json")

# Save state only (pickle - for non-JSON objects)
program.save("program.pkl")

# Save full program with architecture
program.save("program_dir", save_program=True)

# Load state (requires creating instance first)
loaded_program = YourProgramClass()
loaded_program.load("program.json")

# Load full program (no instance needed)
import dspy
loaded_program = dspy.load("program_dir")
```

## Custom Tool Definition

Create tools for agent use with automatic schema extraction.

```python
def get_weather(city: str) -> str:
    """Get current weather information for a city.

    Args:
        city: Name of the city to get weather for

    Returns:
        Weather description string
    """
    return f"The weather in {city} is sunny with a high of 75°F"

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates."""
    import math
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

# Use with ReAct
react = dspy.ReAct(
    "question -> answer",
    tools=[get_weather, calculate_distance],
    max_iters=5
)

result = react(question="What's the weather in New York?")
print(result.answer)
print(result.trajectory)  # Shows tool usage
```

## Adapter Configuration

Choose how DSPy communicates with LMs.

```python
# Chat adapter (default)
adapter = dspy.ChatAdapter(use_native_function_calling=False)
dspy.configure(lm=lm, adapter=adapter)

# JSON adapter (better for structured outputs)
adapter = dspy.JSONAdapter(use_native_function_calling=True)
dspy.configure(lm=lm, adapter=adapter)
```

## Setting LM Per Module

Override global LM for specific modules.

```python
# Global configuration
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Override for specific module
expensive_reasoner = dspy.ChainOfThought("complex_question -> detailed_answer")
expensive_reasoner.set_lm(dspy.LM("openai/gpt-4o"))

# Use in a program
class HybridProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cheap_filter = dspy.Predict("question -> is_complex: bool")
        self.expensive_solver = dspy.ChainOfThought("question -> answer")
        self.expensive_solver.set_lm(dspy.LM("openai/gpt-4o"))

    def forward(self, question):
        if self.cheap_filter(question=question).is_complex:
            return self.expensive_solver(question=question)
        else:
            return dspy.Predict("question -> answer")(question=question)
```

## Complete End-to-End Example

Build, optimize, and deploy a complete AI system.

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2
from dspy.datasets import HotPotQA

# 1. Configure LM
lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_API_KEY")
dspy.configure(lm=lm)

# 2. Define your program
class MultiHopQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_query = dspy.ChainOfThought("question, context -> search_query")
        self.generate_answer = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question):
        # First retrieval
        context1 = self.retrieve(question).passages

        # Generate follow-up query
        search_query = self.generate_query(question=question, context=context1).search_query

        # Second retrieval
        context2 = self.retrieve(search_query).passages

        # Generate final answer
        all_context = context1 + context2
        return self.generate_answer(question=question, context=all_context)

# 3. Load data
dataset = HotPotQA()
trainset = [x.with_inputs('question') for x in dataset.train[:500]]
devset = [x.with_inputs('question') for x in dataset.dev[:200]]

# 4. Define metric
def metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# 5. Configure retriever
colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbert)

# 6. Optimize program
program = MultiHopQA()

optimizer = MIPROv2(metric=metric, auto="medium", num_threads=24)
optimized_program = optimizer.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=3
)

# 7. Evaluate
evaluator = Evaluate(devset=devset, metric=metric, num_threads=16, display_progress=True)
baseline_score = evaluator(program)
optimized_score = evaluator(optimized_program)

print(f"Baseline: {baseline_score:.1%}")
print(f"Optimized: {optimized_score:.1%}")

# 8. Save for production
optimized_program.save("production_model.json")

# 9. Load and use in production
prod_program = MultiHopQA()
prod_program.load("production_model.json")
result = prod_program(question="What castle did David Gregory inherit?")
print(result.answer)
```

---

DSPy excels at building RAG systems, multi-hop question answering, classification tasks, entity extraction, agent-based systems, and complex reasoning pipelines. For RAG applications, developers compose retrieval modules with reasoning modules and optimize the entire pipeline jointly using optimizers like MIPROv2. Multi-hop reasoning systems benefit from DSPy's modular composition, allowing developers to chain query generation, retrieval, and synthesis steps that can be optimized end-to-end. Agent systems leverage dspy.ReAct for tool-using capabilities, enabling LMs to interact with external APIs, databases, and functions while maintaining structured control flow.

The typical integration pattern follows a three-stage workflow: (1) Programming - define task structure using signatures and compose modules; (2) Evaluation - create metrics and collect development sets; (3) Optimization - use DSPy optimizers to automatically improve prompts or fine-tune weights. This workflow enables rapid iteration without manual prompt engineering. Programs save as JSON or pickle files for reproducibility and load back for deployment. DSPy integrates seamlessly with production systems through async operations, streaming, caching, and comprehensive logging. The framework supports all major LLM providers through LiteLLM, making it easy to switch models or use multiple models within the same pipeline. MLflow integration provides experiment tracking, while the modular architecture allows teams to share and reuse optimized components across applications.
