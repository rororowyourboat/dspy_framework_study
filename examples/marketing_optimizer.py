"""Optimize a marketing email generator with DSPy's MIPROv2 teleprompter.

The script walks through a lightweight experimentation workflow:

1. Define signatures and a composable DSPy module for campaign copy.
2. Create a small labeled set of effective marketing briefs.
3. Measure baseline quality with a heuristic metric tailored to growth teams.
4. Compile the program with ``MIPROv2`` to automatically improve prompts.

Usage:
    export OPENAI_API_KEY=...
    python examples/marketing_optimizer.py
"""

from __future__ import annotations

import os

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt.mipro_v2 import MIPROv2


SEGMENT_KEYWORDS = {
    "Revenue marketing managers at B2B SaaS companies": {"pipeline", "ARR", "demo"},
    "Operations leads at omnichannel retailers": {"inventory", "store", "fulfillment"},
    "Field marketers at healthcare technology firms": {"clinic", "patient", "compliance"},
}

GOAL_KEYWORDS = {
    "Increase webinar attendance": {"webinar", "live session", "seat", "register"},
    "Drive product trial sign-ups": {"trial", "experience", "start", "hands-on"},
    "Book sales consultations": {"consult", "strategy call", "schedule", "advisor"},
}

CHANNEL_PHRASES = {
    "email": {"inbox", "email", "newsletter"},
    "linkedin": {"LinkedIn", "social", "feed"},
    "sms": {"text", "SMS", "mobile"},
}


class CampaignPlan(dspy.Signature):
    """Produce a marketing concept with rationale and positioning."""

    customer_segment: str = dspy.InputField()
    product: str = dspy.InputField()
    goal: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="step-by-step thought process for the campaign")
    angle: str = dspy.OutputField(desc="core message angle to pursue")
    talking_points: str = dspy.OutputField(desc="bullet list of talking points")


class ChannelCopy(dspy.Signature):
    """Generate channel-specific copy from a campaign concept."""

    customer_segment: str = dspy.InputField()
    product: str = dspy.InputField()
    goal: str = dspy.InputField()
    preferred_channel: str = dspy.InputField()
    call_to_action: str = dspy.InputField()
    angle: str = dspy.InputField()
    talking_points: str = dspy.InputField()
    subject_line: str = dspy.OutputField(desc="concise subject line or hook")
    channel_copy: str = dspy.OutputField(desc="final copy for the chosen channel")


class GrowthCampaignModule(dspy.Module):
    """Compose reasoning and copy generation for growth marketing teams."""

    def __init__(self) -> None:
        super().__init__()
        self.planner = dspy.ChainOfThought(CampaignPlan)
        self.copywriter = dspy.Predict(ChannelCopy)

    def forward(
        self,
        customer_segment: str,
        product: str,
        goal: str,
        preferred_channel: str,
        call_to_action: str,
    ) -> dspy.Prediction:
        plan = self.planner(
            customer_segment=customer_segment,
            product=product,
            goal=goal,
        )
        creative = self.copywriter(
            customer_segment=customer_segment,
            product=product,
            goal=goal,
            preferred_channel=preferred_channel,
            call_to_action=call_to_action,
            angle=plan.angle,
            talking_points=plan.talking_points,
        )

        return dspy.Prediction(
            reasoning=plan.reasoning,
            angle=plan.angle,
            talking_points=plan.talking_points,
            subject_line=creative.subject_line,
            channel_copy=creative.channel_copy,
        )


def configure_lm() -> dspy.BaseLM:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to run the optimization demo.")

    model = os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    lm = dspy.LM(model=model, api_key=api_key, temperature=0.5, max_tokens=500)
    dspy.configure(lm=lm)
    return lm


def make_example(**kwargs) -> dspy.Example:
    return dspy.Example(**kwargs).with_inputs(
        "customer_segment", "product", "goal", "preferred_channel", "call_to_action"
    )


TRAINSET = [
    make_example(
        customer_segment="Revenue marketing managers at B2B SaaS companies",
        product="Lifecycle analytics platform",
        goal="Increase webinar attendance",
        preferred_channel="email",
        call_to_action="Save your seat",
        subject_line="Unlock pipeline visibility in our live session",
        channel_copy=(
            "Hi Julia,\n\nDemand leaders are telling us it's impossible to steer pipeline without "
            "real-time health alerts. Join our upcoming webinar to see how lifecycle dashboards "
            "keep ARR on track and flag churn risks before QBRs. Save your seat to grab the "
            "automation workbook attendees receive."
        ),
    ),
    make_example(
        customer_segment="Operations leads at omnichannel retailers",
        product="Inventory automation platform",
        goal="Drive product trial sign-ups",
        preferred_channel="linkedin",
        call_to_action="Start your 14-day trial",
        subject_line="Cut stockouts before the holidays",
        channel_copy=(
            "Store operations teams are juggling store and digital stock without the right "
            "alerts. Our automation monitors inventory, triggers replenishment, and syncs "
            "fulfillment in real time. Start your 14-day trial to experience a stress-free peak season."
        ),
    ),
    make_example(
        customer_segment="Field marketers at healthcare technology firms",
        product="Patient engagement suite",
        goal="Book sales consultations",
        preferred_channel="email",
        call_to_action="Book a strategy call",
        subject_line="Help care teams close the loop",
        channel_copy=(
            "Hello Priya,\n\nClinics are asking for HIPAA-compliant follow-up that keeps patients "
            "engaged between visits. We'll walk through how teams like Northbridge Care launch "
            "personalized reminders without risking compliance. Book a strategy call and bring your top questions."
        ),
    ),
]

DEVSET = [
    make_example(
        customer_segment="Revenue marketing managers at B2B SaaS companies",
        product="Lifecycle analytics platform",
        goal="Book sales consultations",
        preferred_channel="email",
        call_to_action="Schedule a strategy session",
        subject_line="",
        channel_copy="",
    ),
    make_example(
        customer_segment="Operations leads at omnichannel retailers",
        product="Inventory automation platform",
        goal="Increase webinar attendance",
        preferred_channel="linkedin",
        call_to_action="Register now",
        subject_line="",
        channel_copy="",
    ),
]


def marketing_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Score copy based on personalization, goal alignment, and CTA clarity."""

    text = pred.channel_copy.lower()
    score = 0.0

    segment_terms = SEGMENT_KEYWORDS.get(example.customer_segment, set())
    if any(term in text for term in segment_terms):
        score += 0.25

    goal_terms = GOAL_KEYWORDS.get(example.goal, set())
    if any(term in text for term in goal_terms):
        score += 0.25

    channel_terms = CHANNEL_PHRASES.get(example.preferred_channel, set())
    if any(term.lower() in text for term in channel_terms):
        score += 0.1

    if example.call_to_action.lower() in text:
        score += 0.25

    subject = getattr(pred, "subject_line", "")
    if subject and len(subject) <= 60:
        score += 0.15

    return min(score, 1.0)


def optimize_module() -> None:
    lm = configure_lm()
    print(f"Configured optimizer with model: {lm.model}")

    baseline_program = GrowthCampaignModule()

    evaluator = Evaluate(devset=DEVSET, metric=marketing_metric, num_threads=4)
    baseline_score = evaluator(baseline_program)
    print(f"Baseline dev score: {baseline_score:.2f}")

    optimizer = MIPROv2(metric=marketing_metric, auto="medium", num_threads=4)
    optimized_program = optimizer.compile(
        baseline_program,
        trainset=TRAINSET,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    optimized_score = evaluator(optimized_program)
    print(f"Optimized dev score: {optimized_score:.2f}")

    live_brief = dspy.Example(
        customer_segment="Field marketers at healthcare technology firms",
        product="Patient engagement suite",
        goal="Increase webinar attendance",
        preferred_channel="email",
        call_to_action="Reserve your seat",
    ).with_inputs(
        "customer_segment", "product", "goal", "preferred_channel", "call_to_action"
    )

    result = optimized_program(**live_brief.inputs().toDict())
    print("\nOptimized copy suggestion:")
    print(f"Subject: {result.subject_line}")
    print(result.channel_copy)


if __name__ == "__main__":
    optimize_module()
