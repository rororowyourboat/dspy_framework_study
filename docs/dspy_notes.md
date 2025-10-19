DSPy exposes a very small API that you can learn quickly. However, building a new AI system is a more open-ended journey of iterative development, in which you compose the tools and design patterns of DSPy to optimize for your objectives. The three stages of building AI systems in DSPy are:

1) DSPy Programming. This is about defining your task, its constraints, exploring a few examples, and using that to inform your initial pipeline design.

2) DSPy Evaluation. Once your system starts working, this is the stage where you collect an initial development set, define your DSPy metric, and use these to iterate on your system more systematically.

3) DSPy Optimization. Once you have a way to evaluate your system, you use DSPy optimizers to tune the prompts or weights in your program.

DSPy is a bet on writing code instead of strings. In other words, building the right control flow is crucial. Start by defining your task. What are the inputs to your system and what should your system produce as output? Is it a chatbot over your data or perhaps a code assistant? Or maybe a system for translation, for highlighting snippets from search results, or for generating reports with citations?

Next, define your initial pipeline. Can your DSPy program just be a single module or do you need to break it down into a few steps? Do you need retrieval or other tools, like a calculator or a calendar API? Is there a typical workflow for solving your problem in multiple well-scoped steps, or do you want more open-ended tool use with agents for your task? Think about these but start simple, perhaps with just a single dspy.ChainOfThought module, then add complexity incrementally based on observations.

As you do this, craft and try a handful of examples of the inputs to your program. Consider using a powerful LM at this point, or a couple of different LMs, just to understand what's possible. Record interesting (both easy and hard) examples you try. This will be useful when you are doing evaluation and optimization later.

The first step in any DSPy code is to set up your language model. For example, you can configure OpenAI's GPT-4o-mini as your default LM as follows.

# Authenticate via `OPENAI_API_KEY` env: import os; os.environ['OPENAI_API_KEY'] = 'here'
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

import dspy
lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='GEMINI_API_KEY')
dspy.configure(lm=lm)

Calling the LM directly.¶
It's easy to call the lm you configured above directly. This gives you a unified API and lets you benefit from utilities like automatic caching.


lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
Using the LM with DSPy modules.¶
Idiomatic DSPy involves using modules, which we discuss in the next guide.


# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)

Using multiple LMs.¶
You can change the default LM globally with dspy.configure or change it inside a block of code with dspy.context.

Tip

Using dspy.configure and dspy.context is thread-safe!


dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-3.5-turbo:', response.answer)
Possible Output:

GPT-4o-mini: The number of floors in the castle David Gregory inherited cannot be determined with the information provided.
GPT-3.5-turbo: The castle David Gregory inherited has 7 floors.
Configuring LM generation.¶
For any LM, you can configure any of the following attributes at initialization or in each subsequent call.


gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
By default LMs in DSPy are cached. If you repeat the same call, you will get the same outputs. But you can turn off caching by setting cache=False.

If you want to keep caching enabled but force a new request (for example, to obtain diverse outputs), pass a unique rollout_id and set a non-zero temperature in your call. DSPy hashes both the inputs and the rollout_id when looking up a cache entry, so different values force a new LM request while still caching future calls with the same inputs and rollout_id. The ID is also recorded in lm.history, which makes it easy to track or compare different rollouts during experiments. Changing only the rollout_id while keeping temperature=0 will not affect the LM's output.


lm("Say this is a test!", rollout_id=1, temperature=1.0)
You can pass these LM kwargs directly to DSPy modules as well. Supplying them at initialization sets the defaults for every call:


predict = dspy.Predict("question -> answer", rollout_id=1, temperature=1.0)
To override them for a single invocation, provide a config dictionary when calling the module:


predict = dspy.Predict("question -> answer")
predict(question="What is 1 + 52?", config={"rollout_id": 5, "temperature": 1.0})
In both cases, rollout_id is forwarded to the underlying LM, affects its caching behavior, and is stored alongside each response so you can replay or analyze specific rollouts later.

Inspecting output and usage metadata.¶
Every LM object maintains the history of its interactions, including inputs, outputs, token usage (and $$$ cost), and metadata.


len(lm.history)  # e.g., 3 calls to the LM

lm.history[-1].keys()  # access the last call to the LM, with all metadata
Output:


dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost', 'timestamp', 'uuid', 'model', 'response_model', 'model_type])
Using the Responses API¶
By default, DSPy calls language models (LMs) using LiteLLM's Chat Completions API, which is suitable for most standard models and tasks. However, some advanced models, such as OpenAI's reasoning models (e.g., gpt-5 or other future models), may offer improved quality or additional features when accessed via the Responses API, which is supported in DSPy.

When should you use the Responses API?

If you are working with models that support or require the responses endpoint (such as OpenAI's reasoning models).
When you want to leverage enhanced reasoning, multi-turn, or richer output capabilities provided by certain models.
How to enable the Responses API in DSPy:

To enable the Responses API, just set model_type="responses" when creating the dspy.LM instance.


import dspy

# Configure DSPy to use the Responses API for your language model
dspy.settings.configure(
    lm=dspy.LM(
        "openai/gpt-5-mini",
        model_type="responses",
        temperature=1.0,
        max_tokens=16000,
    ),
)
Please note that not all models or providers support the Responses API, check LiteLLM's documentation for more details.

Advanced: Building custom LMs and writing your own Adapters.¶
Though rarely needed, you can write custom LMs by inheriting from dspy.BaseLM. Another advanced layer in the DSPy ecosystem is that of adapters, which sit between DSPy signatures and LMs. A future version of this guide will discuss these advanced features, though you likely don't need them.

Signatures¶
When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

A signature is a declarative specification of input/output behavior of a DSPy module. Signatures allow you to tell the LM what it needs to do, rather than specify how we should ask the LM to do it.

You're probably familiar with function signatures, which specify the input and output arguments and their types. DSPy signatures are similar, but with a couple of differences. While typical function signatures just describe things, DSPy Signatures declare and initialize the behavior of modules. Moreover, the field names matter in DSPy Signatures. You express semantic roles in plain English: a question is different from an answer, a sql_query is different from python_code.

Why should I use a DSPy Signature?¶
For modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes). Most people coerce LMs to do tasks by hacking long, brittle prompts. Or by collecting/generating data for fine-tuning. Writing signatures is far more modular, adaptive, and reproducible than hacking at prompts or finetunes. The DSPy compiler will figure out how to build a highly-optimized prompt for your LM (or finetune your small LM) for your signature, on your data, and within your pipeline. In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things and tune the metrics directly.

Inline DSPy Signatures¶
Signatures can be defined as a short string, with argument names and optional types that define semantic roles for inputs/outputs.

Question Answering: "question -> answer", which is equivalent to "question: str -> answer: str" as the default type is always str

Sentiment Classification: "sentence -> sentiment: bool", e.g. True if positive

Summarization: "document -> summary"

Your signatures can also have multiple input/output fields with types:

Retrieval-Augmented Question Answering: "context: list[str], question: str -> answer: str"

Multiple-Choice Question Answering with Reasoning: "question, choices: list[str] -> reasoning: str, selection: int"

Tip: For fields, any valid variable names work! Field names should be semantically meaningful, but start simple and don't prematurely optimize keywords! Leave that kind of hacking to the DSPy compiler. For example, for summarization, it's probably fine to say "document -> summary", "text -> gist", or "long_context -> tldr".

You can also add instructions to your inline signature, which can use variables at runtime. Use the instructions keyword argument to add instructions to your signature.


toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
    )
)
comment = "you are beautiful."
toxicity(comment=comment).toxic
Output:


False
Example A: Sentiment Classification¶

sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
classify(sentence=sentence).sentiment
Output:

True
Example B: Summarization¶

# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)
Possible Output:

The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
Many DSPy modules (except dspy.Predict) return auxiliary information by expanding your signature under the hood.

For example, dspy.ChainOfThought also adds a reasoning field that includes the LM's reasoning before it generates the output summary.


print("Reasoning:", response.reasoning)
Possible Output:

Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
Class-based DSPy Signatures¶
For some advanced tasks, you need more verbose signatures. This is typically to:

Clarify something about the nature of the task (expressed below as a docstring).

Supply hints on the nature of an input field, expressed as a desc keyword argument for dspy.InputField.

Supply constraints on an output field, expressed as a desc keyword argument for dspy.OutputField.

Example C: Classification¶

from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""

    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
classify(sentence=sentence)
Possible Output:

Prediction(
    sentiment='fear'
)
Tip: There's nothing wrong with specifying your requests to the LM more clearly. Class-based Signatures help you with that. However, don't prematurely tune the keywords of your signature by hand. The DSPy optimizers will likely do a better job (and will transfer better across LMs).

Example D: A metric that evaluates faithfulness to citations¶

class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
Possible Output:

Prediction(
    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.",
    faithfulness=False,
    evidence={'goal_count': ["scored twice for the U's"]}
)
Example E: Multi-modal image classification¶

class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
classify(image_1=dspy.Image.from_url(image_url))
Possible Output:


Prediction(
    answer='Labrador Retriever'
)
Type Resolution in Signatures¶
DSPy signatures support various annotation types:

Basic types like str, int, bool
Typing module types like list[str], dict[str, int], Optional[float]. Union[str, int]
Custom types defined in your code
Dot notation for nested types with proper configuration
Special data types like dspy.Image, dspy.History
Working with Custom Types¶

# Simple custom type
class QueryResult(pydantic.BaseModel):
    text: str
    score: float

signature = dspy.Signature("query: str -> result: QueryResult")

class MyContainer:
    class Query(pydantic.BaseModel):
        text: str
    class Score(pydantic.BaseModel):
        score: float

signature = dspy.Signature("query: MyContainer.Query -> score: MyContainer.Score")
Using signatures to build modules & compiling them¶
While signatures are convenient for prototyping with structured inputs/outputs, that's not the only reason to use them!

You should compose multiple signatures into bigger DSPy modules and compile these modules into optimized prompts and finetunes.

Modules¶
A DSPy module is a building block for programs that use LMs.

Each built-in module abstracts a prompting technique (like chain of thought or ReAct). Crucially, they are generalized to handle any signature.

A DSPy module has learnable parameters (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.

How do I use a built-in module, like dspy.Predict or dspy.ChainOfThought?¶
Let's start with the most fundamental module, dspy.Predict. Internally, all other DSPy modules are built using dspy.Predict. We'll assume you are already at least a little familiar with DSPy signatures, which are declarative specs for defining the behavior of any module we use in DSPy.

To use a module, we first declare it by giving it a signature. Then we call the module with the input arguments, and extract the output fields!


sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')

# 2) Call with input argument(s). 
response = classify(sentence=sentence)

# 3) Access the output.
print(response.sentiment)
Output:

True
When we declare a module, we can pass configuration keys to it.

Below, we'll pass n=5 to request five completions. We can also pass temperature or max_len, etc.

Let's use dspy.ChainOfThought. In many cases, simply swapping dspy.ChainOfThought in place of dspy.Predict improves quality.


question = "What's something great about the ColBERT retrieval model?"

# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)

# 2) Call with input argument.
response = classify(question=question)

# 3) Access the outputs.
response.completions.answer
Possible Output:

['One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.',
 'Its ability to efficiently retrieve relevant information from large document collections.',
 'One great thing about the ColBERT retrieval model is its superior performance compared to other models and its efficient use of pre-trained language models.',
 'One great thing about the ColBERT retrieval model is its superior efficiency and accuracy compared to other models.',
 'One great thing about the ColBERT retrieval model is its ability to incorporate user feedback and support complex queries.']
Let's discuss the output object here. The dspy.ChainOfThought module will generally inject a reasoning before the output field(s) of your signature.

Let's inspect the (first) reasoning and answer!


print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
Possible Output:

Reasoning: We can consider the fact that ColBERT has shown to outperform other state-of-the-art retrieval models in terms of efficiency and effectiveness. It uses contextualized embeddings and performs document retrieval in a way that is both accurate and scalable.
Answer: One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.
This is accessible whether we request one or many completions.

We can also access the different completions as a list of Predictions or as several lists, one for each field.


response.completions[3].reasoning == response.completions.reasoning[3]
Output:

True
What other DSPy modules are there? How can I use them?¶
The others are very similar. They mainly change the internal behavior with which your signature is implemented!

dspy.Predict: Basic predictor. Does not modify the signature. Handles the key forms of learning (i.e., storing the instructions and demonstrations and updates to the LM).

dspy.ChainOfThought: Teaches the LM to think step-by-step before committing to the signature's response.

dspy.ProgramOfThought: Teaches the LM to output code, whose execution results will dictate the response.

dspy.ReAct: An agent that can use tools to implement the given signature.

dspy.MultiChainComparison: Can compare multiple outputs from ChainOfThought to produce a final prediction.

We also have some function-style modules:

dspy.majority: Can do basic voting to return the most popular response from a set of predictions.
A few examples of DSPy modules on simple tasks.

Try the examples below after configuring your lm. Adjust the fields to explore what tasks your LM can do well out of the box.


Math
Retrieval-Augmented Generation
Classification
Information Extraction
Agents

def evaluate_math(expression: str) -> float:
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer)
Possible Output:


5761.328

How do I compose multiple modules into a bigger program?¶
DSPy is just Python code that uses modules in any control flow you like, with a little magic internally at compile time to trace your LM calls. What this means is that, you can just call the modules freely.

See tutorials like multi-hop search, whose module is reproduced below as an example.


class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)

        return dspy.Prediction(notes=notes, titles=list(set(titles)))
Then you can create a instance of the custom module class Hop, then invoke it by the __call__ method:


hop = Hop()
print(hop(claim="Stephen Curry is the best 3 pointer shooter ever in the human history"))
How do I track LM usage?¶
Version Requirement

LM usage tracking is available in DSPy version 2.6.16 and later.

DSPy provides built-in tracking of language model usage across all module calls. To enable tracking:


dspy.settings.configure(track_usage=True)
Once enabled, you can access usage statistics from any dspy.Prediction object:


usage = prediction_instance.get_lm_usage()
The usage data is returned as a dictionary that maps each language model name to its usage statistics. Here's a complete example:


import dspy

# Configure DSPy with tracking enabled
dspy.settings.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

# Define a simple program that makes multiple LM calls
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("question, answer -> score")

    def __call__(self, question: str) -> str:
        answer = self.predict1(question=question)
        score = self.predict2(question=question, answer=answer)
        return score

# Run the program and check usage
program = MyProgram()
output = program(question="What is the capital of France?")
print(output.get_lm_usage())
This will output usage statistics like:


{
    'openai/gpt-4o-mini': {
        'completion_tokens': 61,
        'prompt_tokens': 260,
        'total_tokens': 321,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0,
            'audio_tokens': 0,
            'reasoning_tokens': 0,
            'rejected_prediction_tokens': 0,
            'text_tokens': None
        },
        'prompt_tokens_details': {
            'audio_tokens': 0,
            'cached_tokens': 0,
            'text_tokens': None,
            'image_tokens': None
        }
    }
}
When using DSPy's caching features (either in-memory or on-disk via litellm), cached responses won't count toward usage statistics. For example:


# Enable caching
dspy.settings.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

program = MyProgram()

# First call - will show usage statistics
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows token usage

# Second call - same question, will use cache
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows empty dict: {}


---
Understanding DSPy Adapters¶
What are Adapters?¶
Adapters are the bridge between dspy.Predict and the actual Language Model (LM). When you call a DSPy module, the adapter takes your signature, user inputs, and other attributes like demos (few-shot examples) and converts them into multi-turn messages that get sent to the LM.

The adapter system is responsible for:

Translating DSPy signatures into system messages that define the task and request/response structure.
Formatting input data according to the request structure outlined in DSPy signatures.
Parsing LM responses back into structured DSPy outputs, such as dspy.Prediction instances.
Managing conversation history and function calls.
Converting pre-built DSPy types into LM prompt messages, e.g., dspy.Tool, dspy.Image, etc.
Configure Adapters¶
You can use dspy.configure(adapter=...) to choose the adapter for the entire Python process, or with dspy.context(adapter=...): to only affect a certain namespace.

If no adapter is specified in the DSPy workflow, each dspy.Predict.__call__ defaults to using the dspy.ChatAdapter. Thus, the two code snippets below are equivalent:


import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")

import dspy

dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.ChatAdapter(),  # This is the default value
)

predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")
Where Adapters Fit in the System¶
The flow works as follows:

The user calls their DSPy agent, typically a dspy.Module with inputs.
The inner dspy.Predict is invoked to obtain the LM response.
dspy.Predict calls Adapter.format(), which converts its signature, inputs, and demos into multi-turn messages sent to the dspy.LM. dspy.LM is a thin wrapper around litellm, which communicates with the LM endpoint.
The LM receives the messages and generates a response.
Adapter.parse() converts the LM response into structured DSPy outputs, as specified in the signature.
The caller of dspy.Predict receives the parsed outputs.
You can explicitly call Adapter.format() to view the messages sent to the LM.


# Simplified flow example
signature = dspy.Signature("question -> answer")
inputs = {"question": "What is 2+2?"}
demos = [{"question": "What is 1+1?", "answer": "2"}]

adapter = dspy.ChatAdapter()
print(adapter.format(signature, demos, inputs))
The output should resemble:


{'role': 'system', 'content': 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.'}
{'role': 'user', 'content': '[[ ## question ## ]]\nWhat is 1+1?'}
{'role': 'assistant', 'content': '[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n'}
{'role': 'user', 'content': '[[ ## question ## ]]\nWhat is 2+2?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.'}
Types of Adapters¶
DSPy offers several adapter types, each tailored for specific use cases:

ChatAdapter¶
ChatAdapter is the default adapter and works with all language models. It uses a field-based format with special markers.

Format Structure¶
ChatAdapter uses [[ ## field_name ## ]] markers to delineate fields. For fields of non-primitive Python types, it includes the JSON schema of the type. Below, we use dspy.inspect_history() to display the formatted messages by dspy.ChatAdapter clearly.


import dspy
import pydantic

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.ChatAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(dspy.Signature):
    """Get news about the given science field"""

    science_field: str = dspy.InputField()
    year: int = dspy.InputField()
    num_of_outputs: int = dspy.InputField()
    news: list[ScienceNews] = dspy.OutputField(desc="science news")


predict = dspy.Predict(NewsQA)
predict(science_field="Computer Theory", year=2022, num_of_outputs=1)
dspy.inspect_history()
The output looks like:


[2025-08-15T22:24:29.378666]

System message:

Your input fields are:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
Your output fields are:
1. `news` (list[ScienceNews]): science news
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

[[ ## news ## ]]
{news}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "$defs": {"ScienceNews": {"type": "object", "properties": {"scientists_involved": {"type": "array", "items": {"type": "string"}, "title": "Scientists Involved"}, "text": {"type": "string", "title": "Text"}}, "required": ["text", "scientists_involved"], "title": "ScienceNews"}}, "items": {"$ref": "#/$defs/ScienceNews"}}

[[ ## completed ## ]]
In adhering to this structure, your objective is:
        Get news about the given science field


User message:

[[ ## science_field ## ]]
Computer Theory

[[ ## year ## ]]
2022

[[ ## num_of_outputs ## ]]
1

Respond with the corresponding output fields, starting with the field `[[ ## news ## ]]` (must be formatted as a valid Python list[ScienceNews]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## news ## ]]
[
    {
        "scientists_involved": ["John Doe", "Jane Smith"],
        "text": "In 2022, researchers made significant advancements in quantum computing algorithms, demonstrating their potential to solve complex problems faster than classical computers. This breakthrough could revolutionize fields such as cryptography and optimization."
    }
]

[[ ## completed ## ]]
Practice: locate Signature information in the printed LM history

Try adjusting the signature, and observe how the changes are reflected in the printed LM message.

Each field is preceded by a marker [[ ## field_name ## ]]. If an output field has non-primitive types, the instruction includes the type's JSON schema, and the output is formatted accordingly. Because the output field is structured as defined by ChatAdapter, it can be automatically parsed into structured data.

When to Use ChatAdapter¶
ChatAdapter offers the following advantages:

Universal compatibility: Works with all language models, though smaller models may generate responses that do not match the required format.
Fallback protection: If ChatAdapter fails, it automatically retries with JSONAdapter.
In general, ChatAdapter is a reliable choice if you don't have specific requirements.

When Not to Use ChatAdapter¶
Avoid using ChatAdapter if you are:

Latency sensitive: ChatAdapter includes more boilerplate output tokens compared to other adapters, so if you're building a system sensitive to latency, consider using a different adapter.
JSONAdapter¶
JSONAdapter prompts the LM to return JSON data containing all output fields as specified in the signature. It is effective for models that support structured output via the response_format parameter, leveraging native JSON generation capabilities for more reliable parsing.

Format Structure¶
The input part of the prompt formatted by JSONAdapter is similar to ChatAdapter, but the output part differs, as shown below:


import dspy
import pydantic

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(dspy.Signature):
    """Get news about the given science field"""

    science_field: str = dspy.InputField()
    year: int = dspy.InputField()
    num_of_outputs: int = dspy.InputField()
    news: list[ScienceNews] = dspy.OutputField(desc="science news")


predict = dspy.Predict(NewsQA)
predict(science_field="Computer Theory", year=2022, num_of_outputs=1)
dspy.inspect_history()

System message:

Your input fields are:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
Your output fields are:
1. `news` (list[ScienceNews]): science news
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

Outputs will be a JSON object with the following fields.

{
  "news": "{news}        # note: the value you produce must adhere to the JSON schema: {\"type\": \"array\", \"$defs\": {\"ScienceNews\": {\"type\": \"object\", \"properties\": {\"scientists_involved\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"title\": \"Scientists Involved\"}, \"text\": {\"type\": \"string\", \"title\": \"Text\"}}, \"required\": [\"text\", \"scientists_involved\"], \"title\": \"ScienceNews\"}}, \"items\": {\"$ref\": \"#/$defs/ScienceNews\"}}"
}
In adhering to this structure, your objective is:
        Get news about the given science field


User message:

[[ ## science_field ## ]]
Computer Theory

[[ ## year ## ]]
2022

[[ ## num_of_outputs ## ]]
1

Respond with a JSON object in the following order of fields: `news` (must be formatted as a valid Python list[ScienceNews]).


Response:

{
  "news": [
    {
      "text": "In 2022, researchers made significant advancements in quantum computing algorithms, demonstrating that quantum systems can outperform classical computers in specific tasks. This breakthrough could revolutionize fields such as cryptography and complex system simulations.",
      "scientists_involved": [
        "Dr. Alice Smith",
        "Dr. Bob Johnson",
        "Dr. Carol Lee"
      ]
    }
  ]
}
When to Use JSONAdapter¶
JSONAdapter is good at:

Structured output support: When the model supports the response_format parameter.
Low latency: Minimal boilerplate in the LM response results in faster responses.
When Not to Use JSONAdapter¶
Avoid using JSONAdapter if you are:

Using a model that does not natively support structured output, such as a small open-source model hosted on Ollama.
Summary¶
Adapters are a crucial component of DSPy that bridge the gap between structured DSPy signatures and language model APIs. Understanding when and how to use different adapters will help you build more reliable and efficient DSPy programs.

---

Tools¶
DSPy provides powerful support for tool-using agents that can interact with external functions, APIs, and services. Tools enable language models to go beyond text generation by performing actions, retrieving information, and processing data dynamically.

There are two main approaches to using tools in DSPy:

dspy.ReAct - A fully managed tool agent that handles reasoning and tool calls automatically
Manual tool handling - Direct control over tool calls using dspy.Tool, dspy.ToolCalls, and custom signatures
Approach 1: Using dspy.ReAct (Fully Managed)¶
The dspy.ReAct module implements the Reasoning and Acting (ReAct) pattern, where the language model iteratively reasons about the current situation and decides which tools to call.

Basic Example¶

import dspy

# Define your tools as functions
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is sunny and 75°F"

def search_web(query: str) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    return f"Search results for '{query}': [relevant information...]"

# Create a ReAct agent
react_agent = dspy.ReAct(
    signature="question -> answer",
    tools=[get_weather, search_web],
    max_iters=5
)

# Use the agent
result = react_agent(question="What's the weather like in Tokyo?")
print(result.answer)
print("Tool calls made:", result.trajectory)
ReAct Features¶
Automatic reasoning: The model thinks through the problem step by step
Tool selection: Automatically chooses which tool to use based on the situation
Iterative execution: Can make multiple tool calls to gather information
Error handling: Built-in error recovery for failed tool calls
Trajectory tracking: Complete history of reasoning and tool calls
ReAct Parameters¶

react_agent = dspy.ReAct(
    signature="question -> answer",  # Input/output specification
    tools=[tool1, tool2, tool3],     # List of available tools
    max_iters=10                     # Maximum number of tool call iterations
)
Approach 2: Manual Tool Handling¶
For more control over the tool calling process, you can manually handle tools using DSPy's tool types.

Basic Setup¶

import dspy

class ToolSignature(dspy.Signature):
    """Signature for manual tool handling."""
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    outputs: dspy.ToolCalls = dspy.OutputField()

def weather(city: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {city} is sunny"

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Note: Use safely in production
        return f"The result is {result}"
    except:
        return "Invalid expression"

# Create tool instances
tools = {
    "weather": dspy.Tool(weather),
    "calculator": dspy.Tool(calculator)
}

# Create predictor
predictor = dspy.Predict(ToolSignature)

# Make a prediction
response = predictor(
    question="What's the weather in New York?",
    tools=list(tools.values())
)

# Execute the tool calls
for call in response.outputs.tool_calls:
    # Execute the tool call
    result = call.execute()
    print(f"Tool: {call.name}")
    print(f"Args: {call.args}")
    print(f"Result: {result}")
Understanding dspy.Tool¶
The dspy.Tool class wraps regular Python functions to make them compatible with DSPy's tool system:


def my_function(param1: str, param2: int = 5) -> str:
    """A sample function with parameters."""
    return f"Processed {param1} with value {param2}"

# Create a tool
tool = dspy.Tool(my_function)

# Tool properties
print(tool.name)        # "my_function"
print(tool.desc)        # The function's docstring
print(tool.args)        # Parameter schema
print(str(tool))        # Full tool description
Understanding dspy.ToolCalls¶
The dspy.ToolCalls type represents the output from a model that can make tool calls. Each individual tool call can be executed using the execute method:


# After getting a response with tool calls
for call in response.outputs.tool_calls:
    print(f"Tool name: {call.name}")
    print(f"Arguments: {call.args}")

    # Execute individual tool calls with different options:

    # Option 1: Automatic discovery (finds functions in locals/globals)
    result = call.execute()  # Automatically finds functions by name

    # Option 2: Pass tools as a dict (most explicit)
    result = call.execute(functions={"weather": weather, "calculator": calculator})

    # Option 3: Pass Tool objects as a list
    result = call.execute(functions=[dspy.Tool(weather), dspy.Tool(calculator)])

    print(f"Result: {result}")
Using Native Tool Calling¶
DSPy adapters support native function calling, which leverages the underlying language model's built-in tool calling capabilities rather than relying on text-based parsing. This approach can provide more reliable tool execution and better integration with models that support native function calling.

Native tool calling doesn't guarantee better quality

It's possible that native tool calling produces lower quality than custom tool calling.

Adapter Behavior¶
Different DSPy adapters have different defaults for native function calling:

ChatAdapter - Uses use_native_function_calling=False by default (relies on text parsing)
JSONAdapter - Uses use_native_function_calling=True by default (uses native function calling)
You can override these defaults by explicitly setting the use_native_function_calling parameter when creating an adapter.

Configuration¶

import dspy

# ChatAdapter with native function calling enabled
chat_adapter_native = dspy.ChatAdapter(use_native_function_calling=True)

# JSONAdapter with native function calling disabled
json_adapter_manual = dspy.JSONAdapter(use_native_function_calling=False)

# Configure DSPy to use the adapter
dspy.configure(lm=dspy.LM(model="openai/gpt-4o"), adapter=chat_adapter_native)
You can enable the MLflow tracing to check how native tool calling is being used. If you use JSONAdapter or ChatAdapter with native function calling enabled on the code snippet as provided in the section above, you should see native function calling arg tools is set like the screenshot below:

native tool calling

Model Compatibility¶
Native function calling automatically detects model support using litellm.supports_function_calling(). If the model doesn't support native function calling, DSPy will fall back to manual text-based parsing even when use_native_function_calling=True is set.

Best Practices¶
1. Tool Function Design¶
Clear docstrings: Tools work better with descriptive documentation
Type hints: Provide clear parameter and return types
Simple parameters: Use basic types (str, int, bool, dict, list) or Pydantic models

def good_tool(city: str, units: str = "celsius") -> str:
    """
    Get weather information for a specific city.

    Args:
        city: The name of the city to get weather for
        units: Temperature units, either 'celsius' or 'fahrenheit'

    Returns:
        A string describing the current weather conditions
    """
    # Implementation with proper error handling
    if not city.strip():
        return "Error: City name cannot be empty"

    # Weather logic here...
    return f"Weather in {city}: 25°{units[0].upper()}, sunny"
2. Choosing Between ReAct and Manual Handling¶
Use dspy.ReAct when:

You want automatic reasoning and tool selection
The task requires multiple tool calls
You need built-in error recovery
You want to focus on tool implementation rather than orchestration
Use manual tool handling when:

You need precise control over tool execution
You want custom error handling logic
You want to minimize the latency
Your tool returns nothing (void function)
Tools in DSPy provide a powerful way to extend language model capabilities beyond text generation. Whether using the fully automated ReAct approach or manual tool handling, you can build sophisticated agents that interact with the world through code.

---
Model Context Protocol (MCP)¶
The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to language models. DSPy supports MCP, allowing you to use tools from any MCP server with DSPy agents.

Installation¶
Install DSPy with MCP support:


pip install -U "dspy[mcp]"
Overview¶
MCP enables you to:

Use standardized tools - Connect to any MCP-compatible server.
Share tools across stacks - Use the same tools across different frameworks.
Simplify integration - Convert MCP tools to DSPy tools with one line.
DSPy does not handle MCP server connections directly. You can use client interfaces of the mcp library to establish the connection and pass mcp.ClientSession to dspy.Tool.from_mcp_tool in order to convert mcp tools into DSPy tools.

Using MCP with DSPy¶
1. HTTP Server (Remote)¶
For remote MCP servers over HTTP, use the streamable HTTP transport:


import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    # Connect to HTTP MCP server
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List and convert tools
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Create and use ReAct agent
            class TaskSignature(dspy.Signature):
                task: str = dspy.InputField()
                result: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=TaskSignature,
                tools=dspy_tools,
                max_iters=5
            )

            result = await react_agent.acall(task="Check the weather in Tokyo")
            print(result.result)

asyncio.run(main())
2. Stdio Server (Local Process)¶
The most common way to use MCP is with a local server process communicating via stdio:


import asyncio
import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Configure the stdio server
    server_params = StdioServerParameters(
        command="python",                    # Command to run
        args=["path/to/your/mcp_server.py"], # Server script path
        env=None,                            # Optional environment variables
    )

    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            response = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Create a ReAct agent with the tools
            class QuestionAnswer(dspy.Signature):
                """Answer questions using available tools."""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=QuestionAnswer,
                tools=dspy_tools,
                max_iters=5
            )

            # Use the agent
            result = await react_agent.acall(
                question="What is 25 + 17?"
            )
            print(result.answer)

# Run the async function
asyncio.run(main())
Tool Conversion¶
DSPy automatically handles the conversion from MCP tools to DSPy tools:


# MCP tool from session
mcp_tool = response.tools[0]

# Convert to DSPy tool
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)

# The DSPy tool preserves:
# - Tool name and description
# - Parameter schemas and types
# - Argument descriptions
# - Async execution support

# Use it like any DSPy tool
result = await dspy_tool.acall(param1="value", param2=123)
Learn More¶
MCP Official Documentation
MCP Python SDK
DSPy MCP Tutorial
DSPy Tools Documentation
MCP integration in DSPy makes it easy to use standardized tools from any MCP server, enabling powerful agent capabilities with minimal setup.

---

Evaluation in DSPy¶
Once you have an initial system, it's time to collect an initial development set so you can refine it more systematically. Even 20 input examples of your task can be useful, though 200 goes a long way. Depending on your metric, you either just need inputs and no labels at all, or you need inputs and the final outputs of your system. (You almost never need labels for the intermediate steps in your program in DSPy.) You can probably find datasets that are adjacent to your task on, say, HuggingFace datasets or in a naturally occurring source like StackExchange. If there's data whose licenses are permissive enough, we suggest you use them. Otherwise, you can label a few examples by hand or start deploying a demo of your system and collect initial data that way.

Next, you should define your DSPy metric. What makes outputs from your system good or bad? Invest in defining metrics and improving them incrementally over time; it's hard to consistently improve what you aren't able to define. A metric is a function that takes examples from your data and takes the output of your system, and returns a score. For simple tasks, this could be just "accuracy", e.g. for simple classification or short-form QA tasks. For most applications, your system will produce long-form outputs, so your metric will be a smaller DSPy program that checks multiple properties of the output. Getting this right on the first try is unlikely: start with something simple and iterate.

Now that you have some data and a metric, run development evaluations on your pipeline designs to understand their tradeoffs. Look at the outputs and the metric scores. This will probably allow you to spot any major issues, and it will define a baseline for your next steps.

---
Data¶
DSPy is a machine learning framework, so working in it involves training sets, development sets, and test sets. For each example in your data, we distinguish typically between three types of values: the inputs, the intermediate labels, and the final label. You can use DSPy effectively without any intermediate or final labels, but you will need at least a few example inputs.

DSPy Example objects¶
The core data type for data in DSPy is Example. You will use Examples to represent items in your training set and test set.

DSPy Examples are similar to Python dicts but have a few useful utilities. Your DSPy modules will return values of the type Prediction, which is a special sub-class of Example.

When you use DSPy, you will do a lot of evaluation and optimization runs. Your individual datapoints will be of type Example:


qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
Output:

Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
This is a question?
This is an answer.
Examples can have any field keys and any value types, though usually values are strings.


object = Example(field1=value1, field2=value2, field3=value3, ...)
You can now express your training set for example as:


trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
Specifying Input Keys¶
In traditional ML, there are separated "inputs" and "labels".

In DSPy, the Example objects have a with_inputs() method, which can mark specific fields as inputs. (The rest are just metadata or labels.)


# Single Input.
print(qa_pair.with_inputs("question"))

# Multiple Inputs; be careful about marking your labels as inputs unless you mean it.
print(qa_pair.with_inputs("question", "answer"))
Values can be accessed using the .(dot) operator. You can access the value of key name in defined object Example(name="John Doe", job="sleep") through object.name.

To access or exclude certain keys, use inputs() and labels() methods to return new Example objects containing only input or non-input keys, respectively.


article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)
Output


Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys={'article'})
Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)

---

Metrics¶
DSPy is a machine learning framework, so you must think about your automatic metrics for evaluation (to track your progress) and optimization (so DSPy can make your programs more effective).

What is a metric and how do I define a metric for my task?¶
A metric is just a function that will take examples from your data and the output of your system and return a score that quantifies how good the output is. What makes outputs from your system good or bad?

For simple tasks, this could be just "accuracy" or "exact match" or "F1 score". This may be the case for simple classification or short-form QA tasks.

However, for most applications, your system will output long-form outputs. There, your metric should probably be a smaller DSPy program that checks multiple properties of the output (quite possibly using AI feedback from LMs).

Getting this right on the first try is unlikely, but you should start with something simple and iterate.

Simple metrics¶
A DSPy metric is just a function in Python that takes example (e.g., from your training or dev set) and the output pred from your DSPy program, and outputs a float (or int or bool) score.

Your metric should also accept an optional third argument called trace. You can ignore this for a moment, but it will enable some powerful tricks if you want to use your metric for optimization.

Here's a simple example of a metric that's comparing example.answer and pred.answer. This particular metric will return a bool.


def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
Some people find these utilities (built-in) convenient:

dspy.evaluate.metrics.answer_exact_match
dspy.evaluate.metrics.answer_passage_match
Your metrics could be more complex, e.g. check for multiple properties. The metric below will return a float if trace is None (i.e., if it's used for evaluation or optimization), and will return a bool otherwise (i.e., if it's used to bootstrap demonstrations).


def validate_context_and_answer(example, pred, trace=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None: # if we're doing evaluation or optimization
        return (answer_match + context_match) / 2.0
    else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
        return answer_match and context_match
Defining a good metric is an iterative process, so doing some initial evaluations and looking at your data and outputs is key.

Evaluation¶
Once you have a metric, you can run evaluations in a simple Python loop.


scores = []
for x in devset:
    pred = program(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
If you need some utilities, you can also use the built-in Evaluate utility. It can help with things like parallel evaluation (multiple threads) or showing you a sample of inputs/outputs and the metric scores.


from dspy.evaluate import Evaluate

# Set up the evaluator, which can be re-used in your code.
evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1, display_progress=True, display_table=5)

# Launch evaluation.
evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)
Intermediate: Using AI feedback for your metric¶
For most applications, your system will output long-form outputs, so your metric should check multiple dimensions of the output using AI feedback from LMs.

This simple signature could come in handy.


# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
For example, below is a simple metric that checks a generated tweet (1) answers a given question correctly and (2) whether it's also engaging. We also check that (3) len(tweet) <= 280 characters.


def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"

    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None: return score >= 2
    return score / 2.0
When compiling, trace is not None, and we want to be strict about judging things, so we will only return True if score >= 2. Otherwise, we return a score out of 1.0 (i.e., score / 2.0).

Advanced: Using a DSPy program as your metric¶
If your metric is itself a DSPy program, one of the most powerful ways to iterate is to compile (optimize) your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5) so the metric's metric is easy to define and optimize by collecting a few examples.

Advanced: Accessing the trace¶
When your metric is used during evaluation runs, DSPy will not try to track the steps of your program.

But during compiling (optimization), DSPy will trace your LM calls. The trace will contain inputs/outputs to each DSPy predictor and you can leverage that to validate intermediate steps for optimization.


def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True


---

Optimization in DSPy¶
Once you have a system and a way to evaluate it, you can use DSPy optimizers to tune the prompts or weights in your program. Now it's useful to expand your data collection effort into building a training set and a held-out test set, in addition to the development set you've been using for exploration. For the training set (and its subset, validation set), you can often get substantial value out of 30 examples, but aim for at least 300 examples. Some optimizers accept a trainset only. Others ask for a trainset and a valset. When splitting data for most prompt optimizers, we recommend an unusual split compared to deep neural networks: 20% for training, 80% for validation. This reverse allocation emphasizes stable validation, since prompt-based optimizers often overfit to small training sets. In contrast, the dspy.GEPA optimizer follows the more standard ML convention: Maximize the training set size, while keeping the validation set just large enough to reflect the distribution of the downstream tasks (test set).

After your first few optimization runs, you are either very happy with everything or you've made a lot of progress but you don't like something about the final program or the metric. At this point, go back to step 1 (Programming in DSPy) and revisit the major questions. Did you define your task well? Do you need to collect (or find online) more data for your problem? Do you want to update your metric? And do you want to use a more sophisticated optimizer? Do you need to consider advanced features like DSPy Assertions? Or, perhaps most importantly, do you want to add some more complexity or steps in your DSPy program itself? Do you want to use multiple optimizers in a sequence?

Iterative development is key. DSPy gives you the pieces to do that incrementally: iterating on your data, your program structure, your metric, and your optimization steps. Optimizing complex LM programs is an entirely new paradigm that only exists in DSPy at the time of writing (update: there are now numerous DSPy extension frameworks, so this part is no longer true :-), so naturally the norms around what to do are still emerging. If you need help, we recently created a Discord server for the community.

---
DSPy Optimizers (formerly Teleprompters)¶
A DSPy optimizer is an algorithm that can tune the parameters of a DSPy program (i.e., the prompts and/or the LM weights) to maximize the metrics you specify, like accuracy.

A typical DSPy optimizer takes three things:

Your DSPy program. This may be a single module (e.g., dspy.Predict) or a complex multi-module program.

Your metric. This is a function that evaluates the output of your program, and assigns it a score (higher is better).

A few training inputs. This may be very small (i.e., only 5 or 10 examples) and incomplete (only inputs to your program, without any labels).

If you happen to have a lot of data, DSPy can leverage that. But you can start small and get strong results.

Note: Formerly called teleprompters. We are making an official name update, which will be reflected throughout the library and documentation.

What does a DSPy Optimizer tune? How does it tune them?¶
Different optimizers in DSPy will tune your program's quality by synthesizing good few-shot examples for every module, like dspy.BootstrapRS,1 proposing and intelligently exploring better natural-language instructions for every prompt, like dspy.MIPROv2,2 and dspy.GEPA,3 and building datasets for your modules and using them to finetune the LM weights in your system, like dspy.BootstrapFinetune.4

What's an example of a DSPy optimizer? How do different optimizers work?
What DSPy Optimizers are currently available?¶
Optimizers can be accessed via from dspy.teleprompt import *.

Automatic Few-Shot Learning¶
These optimizers extend the signature by automatically generating and including optimized examples within the prompt sent to the model, implementing few-shot learning.

LabeledFewShot: Simply constructs few-shot examples (demos) from provided labeled input and output data points. Requires k (number of examples for the prompt) and trainset to randomly select k examples from.

BootstrapFewShot: Uses a teacher module (which defaults to your program) to generate complete demonstrations for every stage of your program, along with labeled examples in trainset. Parameters include max_labeled_demos (the number of demonstrations randomly selected from the trainset) and max_bootstrapped_demos (the number of additional examples generated by the teacher). The bootstrapping process employs the metric to validate demonstrations, including only those that pass the metric in the "compiled" prompt. Advanced: Supports using a teacher program that is a different DSPy program that has compatible structure, for harder tasks.

BootstrapFewShotWithRandomSearch: Applies BootstrapFewShot several times with random search over generated demonstrations, and selects the best program over the optimization. Parameters mirror those of BootstrapFewShot, with the addition of num_candidate_programs, which specifies the number of random programs evaluated over the optimization, including candidates of the uncompiled program, LabeledFewShot optimized program, BootstrapFewShot compiled program with unshuffled examples and num_candidate_programs of BootstrapFewShot compiled programs with randomized example sets.

KNNFewShot. Uses k-Nearest Neighbors algorithm to find the nearest training example demonstrations for a given input example. These nearest neighbor demonstrations are then used as the trainset for the BootstrapFewShot optimization process.

Automatic Instruction Optimization¶
These optimizers produce optimal instructions for the prompt and, in the case of MIPROv2 can also optimize the set of few-shot demonstrations.

COPRO: Generates and refines new instructions for each step, and optimizes them with coordinate ascent (hill-climbing using the metric function and the trainset). Parameters include depth which is the number of iterations of prompt improvement the optimizer runs over.

MIPROv2: Generates instructions and few-shot examples in each step. The instruction generation is data-aware and demonstration-aware. Uses Bayesian Optimization to effectively search over the space of generation instructions/demonstrations across your modules.

SIMBA

GEPA: Uses LM's to reflect on the DSPy program's trajectory, to identify what worked, what didn't and propose prompts addressing the gaps. Additionally, GEPA can leverage domain-specific textual feedback to rapidly improve the DSPy program. Detailed tutorials on using GEPA are available at dspy.GEPA Tutorials.

Automatic Finetuning¶
This optimizer is used to fine-tune the underlying LLM(s).

BootstrapFinetune: Distills a prompt-based DSPy program into weight updates. The output is a DSPy program that has the same steps, but where each step is conducted by a finetuned model instead of a prompted LM. See the classification fine-tuning tutorial for a complete example.
Program Transformations¶
Ensemble: Ensembles a set of DSPy programs and either uses the full set or randomly samples a subset into a single program.
Which optimizer should I use?¶
Ultimately, finding the ‘right’ optimizer to use & the best configuration for your task will require experimentation. Success in DSPy is still an iterative process - getting the best performance on your task will require you to explore and iterate.

That being said, here's the general guidance on getting started:

If you have very few examples (around 10), start with BootstrapFewShot.
If you have more data (50 examples or more), try BootstrapFewShotWithRandomSearch.
If you prefer to do instruction optimization only (i.e. you want to keep your prompt 0-shot), use MIPROv2 configured for 0-shot optimization.
If you’re willing to use more inference calls to perform longer optimization runs (e.g. 40 trials or more), and have enough data (e.g. 200 examples or more to prevent overfitting) then try MIPROv2.
If you have been able to use one of these with a large LM (e.g., 7B parameters or above) and need a very efficient program, finetune a small LM for your task with BootstrapFinetune.
How do I use an optimizer?¶
They all share this general interface, with some differences in the keyword arguments (hyperparameters). A full list can be found in the API reference.

Let's see this with the most common one, BootstrapFewShotWithRandomSearch.


from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
# The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)

teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config)
optimized_program = teleprompter.compile(YOUR_PROGRAM_HERE, trainset=YOUR_TRAINSET_HERE)
Getting Started III: Optimizing the LM prompts or weights in DSPy programs

A typical simple optimization run costs on the order of $2 USD and takes around ten minutes, but be careful when running optimizers with very large LMs or very large datasets. Optimizer runs can cost as little as a few cents or up to tens of dollars, depending on your LM, dataset, and configuration.


Optimizing prompts for a ReAct agent
Optimizing prompts for RAG
Optimizing weights for Classification
This is a minimal but fully runnable example of setting up a dspy.ReAct agent that answers questions via search from Wikipedia and then optimizing it using dspy.MIPROv2 in the cheap light mode on 500 question-answer pairs sampled from the HotPotQA dataset.


import dspy
from dspy.datasets import HotPotQA

dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

def search(query: str) -> list[str]:
    """Retrieves abstracts from Wikipedia."""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
react = dspy.ReAct("question -> answer", tools=[search])

tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
optimized_react = tp.compile(react, trainset=trainset)
An informal run similar to this on DSPy 2.5.29 raises ReAct's score from 24% to 51%.


Saving and loading optimizer output¶
After running a program through an optimizer, it's useful to also save it. At a later point, a program can be loaded from a file and used for inference. For this, the load and save methods can be used.


optimized_program.save(YOUR_SAVE_PATH)
The resulting file is in plain-text JSON format. It contains all the parameters and steps in the source program. You can always read it and see what the optimizer generated.

To load a program from a file, you can instantiate an object from that class and then call the load method on it.


loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load(path=YOUR_SAVE_PATH)

---
