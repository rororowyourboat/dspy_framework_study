# %%
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_value")
# %%
import dspy # type: ignore
# %%
# configuring dspy
lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
dspy.configure(lm=lm, verbose=True, )
# %%

math = dspy.ChainOfThought("question -> answer: float")
print(math(question="Two dice are tossed. What is the probability that the sum equals two?"))
# %%
# def search_wikipedia(query: str) -> list[str]:
#     results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
#     return [x["text"] for x in results]

# rag = dspy.ChainOfThought("context, question -> response")

# question = "What's the name of the castle that David Gregory inherited?"
# rag(context=search_wikipedia(question), question=question)
# %%
from typing import Literal

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Classify)
print(classify(sentence="This book was super fun to read, though not the last chapter."))
# %%

class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

module = dspy.Predict(ExtractInfo)

text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."
response = module(text=text)

print(response.title)
print(response.headings)
print(response.entities)
# %%

# def evaluate_math(expression: str):
#     return dspy.PythonInterpreter({}).execute(expression)

# def search_wikipedia(query: str):
#     results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
#     return [x["text"] for x in results]

# react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

# pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
# print(pred.answer)
# %%
class Outline(dspy.Signature):
    """Outline a thorough overview of a topic."""

    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")

class DraftSection(dspy.Signature):
    """Draft a top-level section of an article."""

    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section")

class DraftArticle(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
            section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)

draft_article = DraftArticle()
article = draft_article(topic="World Cup 2002")
# %%
for section in article.sections:
    print(section)
    print()
# %%