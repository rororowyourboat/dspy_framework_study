"""Use DSPy's ReAct agent to combine reasoning with marketing-specific tools.

The script wires three lightweight Python utilities into a ReAct agent:

* ``competitor_briefing`` surfaces static insights about competing brands.
* ``roi_estimator`` evaluates spend, lead, and conversion assumptions.
* ``channel_reference`` summarizes historical channel performance.

Together they show how DSPy agents can orchestrate proprietary marketing
knowledge with large language models.

Usage:
    export OPENAI_API_KEY=...
    python examples/react_campaign_assistant.py
"""

from __future__ import annotations

import os
import re

import dspy


COMPETITOR_DATA = {
    "acme analytics": (
        "Acme Analytics recently launched an AI-based forecasting suite that emphasizes "
        "predictive churn alerts and budget pacing dashboards. Their latest messaging "
        "focuses on \"own your pipeline outcomes\" and ROI accountability for demand teams."
    ),
    "northstar metrics": (
        "Northstar Metrics is leaning into RevOps automation with templates for "
        "executive scorecards, highlighting that customers see 18% faster campaign "
        "turnaround and stronger board reporting."
    ),
}

CHANNEL_HISTORY = {
    "webinar": "Webinars drive the highest ACV opportunities but require 3-week prep cycles.",
    "linkedin": "LinkedIn ads deliver efficient top-of-funnel leads with 2.4% CTR.",
    "email": "Email nurture drips convert warm webinar attendees at 22% within 14 days.",
}


def competitor_briefing(query: str) -> str:
    """Return canned competitor intel based on a fuzzy keyword search."""

    normalized = query.lower()
    for name, summary in COMPETITOR_DATA.items():
        if name in normalized:
            return summary
    return "No stored briefing for that competitor."


def roi_estimator(assumptions: str) -> str:
    """Calculate a simple ROI summary from spend, leads, and conversions."""

    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", assumptions)]
    if len(numbers) < 3:
        return "Provide spend, total leads, and closed-won counts to estimate ROI."

    spend, leads, wins = numbers[:3]
    acv = numbers[3] if len(numbers) > 3 else 0.0
    cpl = spend / leads if leads else 0.0
    cac = spend / wins if wins else 0.0
    revenue = wins * acv
    roi = (revenue - spend) / spend if spend else 0.0

    return (
        f"Spend ${spend:,.0f} for {leads:.0f} leads and {wins:.0f} wins. "
        f"Cost per lead: ${cpl:,.2f}. Customer acquisition cost: ${cac:,.2f}. "
        f"Projected revenue: ${revenue:,.0f}. ROI: {roi:.1%}."
    )


def channel_reference(channel_query: str) -> str:
    """Look up a stored insight about a marketing channel."""

    normalized = channel_query.lower()
    for channel, detail in CHANNEL_HISTORY.items():
        if channel in normalized:
            return detail
    return "No channel insight recorded."


def configure_lm() -> dspy.BaseLM:
    """Configure the default language model for the agent."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the agent demo.")

    model = os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    lm = dspy.LM(model=model, api_key=api_key, temperature=0.2, max_tokens=400)
    dspy.configure(lm=lm)
    return lm


def run_agent(question: str) -> dspy.Prediction:
    """Answer a marketing planning question with tool use."""

    agent = dspy.ReAct(
        "question -> answer, confidence: float",
        tools=[competitor_briefing, roi_estimator, channel_reference],
        max_iters=6,
    )
    return agent(question=question)


def main() -> None:
    lm = configure_lm()
    print(f"Configured ReAct agent with model: {lm.model}")

    question = (
        "We are planning a webinar campaign to win accounts from Acme Analytics. "
        "Use the available tools to summarize Acme's current positioning, estimate ROI "
        "if we invest $15000 for 450 leads that close 75 deals at $399 each, and confirm "
        "which follow-up channel historically nurtures webinar leads best. "
        "Finish with a confident recommendation."
    )

    answer = run_agent(question)
    print("\nAgent response:")
    print(answer.answer)
    print(f"\nConfidence: {answer.confidence:.0%}")

    if hasattr(answer, "trajectory"):
        print("\nReAct trajectory:")
        for step in answer.trajectory:
            print(f"- {step}")


if __name__ == "__main__":
    main()
