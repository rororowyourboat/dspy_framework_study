"""Demonstrate core DSPy abstractions with a simple marketing scenario.

The script highlights four foundational concepts:

1. **Signatures** describe inputs and outputs for structured model calls.
2. **Predict** modules map those signatures directly to model invocations.
3. **ChainOfThought** reasoning enables richer intermediate explanations.
4. **Custom DSPy modules** compose multiple building blocks into
   higher-level marketing assistants.

Usage:
    export OPENAI_API_KEY=...  # Your model provider key
    python examples/basic_showcase.py

The script defaults to ``openai/gpt-4o-mini`` but you can override the
model by setting ``DSPY_MODEL``.
"""

from __future__ import annotations

import os

import dspy


class SentimentSignature(dspy.Signature):
    """Classify the overall sentiment of a customer quote."""

    sentence: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, neutral, or negative sentiment label")


def configure_lm() -> dspy.BaseLM:
    """Configure a language model for DSPy using environment variables."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable to run the examples.")

    model = os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    lm = dspy.LM(model=model, api_key=api_key, max_tokens=256, temperature=0.3)
    dspy.configure(lm=lm)
    return lm


def run_predict_module() -> dspy.Prediction:
    """Classify sentiment for a direct customer quote."""

    sentiment_classifier = dspy.Predict(SentimentSignature)
    return sentiment_classifier(
        sentence="I love how quickly the onboarding emails helped me understand the product!"
    )


def run_chain_of_thought() -> dspy.Prediction:
    """Produce a marketing angle with explicit reasoning steps."""

    persona_reasoner = dspy.ChainOfThought(
        "persona -> reasoning, campaign_angle",
        temperature=0.4,
    )
    return persona_reasoner(
        persona=(
            "Time-strapped operations managers at mid-market retailers who need automation "
            "to reduce manual reporting"
        )
    )


class CampaignPlanner(dspy.Module):
    """Compose DSPy modules to turn a persona into an email campaign."""

    def __init__(self) -> None:
        super().__init__()
        self.idea_generator = dspy.ChainOfThought(
            "product, persona -> reasoning, campaign_theme"
        )
        self.copy_generator = dspy.Predict(
            "campaign_theme, product -> subject_line, email_body"
        )

    def forward(self, product: str, persona: str) -> dspy.Prediction:
        idea = self.idea_generator(product=product, persona=persona)
        copy = self.copy_generator(campaign_theme=idea.campaign_theme, product=product)

        return dspy.Prediction(
            reasoning=idea.reasoning,
            campaign_theme=idea.campaign_theme,
            subject_line=copy.subject_line,
            email_body=copy.email_body,
        )


def run_program_module() -> dspy.Prediction:
    """Generate a themed nurture email for a marketing team."""

    planner = CampaignPlanner()
    return planner(
        product="Workflow automation platform with real-time dashboards",
        persona=(
            "Revenue marketing directors at SaaS companies seeking stronger lifecycle analytics"
        ),
    )


def main() -> None:
    lm = configure_lm()
    print(f"Configured DSPy with model: {lm.model}")

    sentiment = run_predict_module()
    print("\nSentiment classification:")
    print(f"  Label: {sentiment.sentiment}\n")

    cot = run_chain_of_thought()
    print("Persona reasoning with Chain-of-Thought:")
    print(f"  Reasoning: {cot.reasoning}")
    print(f"  Campaign angle: {cot.campaign_angle}\n")

    campaign = run_program_module()
    print("Composed marketing campaign module:")
    print(f"  Theme: {campaign.campaign_theme}")
    print(f"  Subject line: {campaign.subject_line}")
    print("  Email body:\n" + campaign.email_body)


if __name__ == "__main__":
    main()
