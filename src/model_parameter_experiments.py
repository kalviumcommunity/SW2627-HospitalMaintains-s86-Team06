import os
from typing import Dict, List


PROMPT = (
    "You are a factual hospital assistant. Using the hospital guideline, "
    "answer: 'What is the patient refund window?' Answer in one sentence."
)


def mock_temperature_response(temperature: float) -> str:
    """Return a deterministic demo answer for a given temperature setting."""
    if temperature <= 0.2:
        return "The hospital refund window is 30 days from the date of purchase, with proof of purchase required."
    if temperature <= 0.7:
        return "Based on the policy, patients generally have a 30-day refund window from purchase, provided they keep their receipt and the item qualifies."
    return (
        "The hospital policy is designed to be patient-friendly: in many cases, a customer can request a refund "
        "within about 30 days, though some items and services may vary depending on the specific terms and supporting evidence."
    )


def mock_max_tokens_response(max_tokens: int) -> str:
    """Generate a response whose token-like length is limited by the requested budget."""
    base = (
        "The hospital refund window is 30 days from the date of purchase. "
        "Proof of purchase is required for a valid refund request."
    )
    words = base.split()
    return " ".join(words[:max_tokens])


def mock_stop_response(stop: str = "###") -> str:
    """Simulate stop behavior by returning text up to the stop token marker."""
    text = (
        "The hospital refund window is 30 days from the date of purchase. "
        "Proof of purchase is required. ### Additional policy notes are not included."
    )
    return text.split(stop)[0].strip()


def run_temperature_experiment() -> Dict[str, str]:
    experiments = {}
    for temp in [0.0, 0.4, 1.0]:
        experiments[f"temperature={temp}"] = mock_temperature_response(temp)
    return experiments


def run_max_tokens_experiment() -> Dict[str, str]:
    return {
        "max_tokens=20": mock_max_tokens_response(20),
        "max_tokens=12": mock_max_tokens_response(12),
        "max_tokens=6": mock_max_tokens_response(6),
    }


def run_stop_experiment() -> Dict[str, str]:
    return {
        "stop=###": mock_stop_response("###"),
        "stop=END": mock_stop_response("END"),
    }


def print_experiment_summary() -> None:
    print("=== Temperature experiment ===")
    for label, answer in run_temperature_experiment().items():
        print(f"{label}: {answer}")
    print()

    print("=== max_tokens experiment ===")
    for label, answer in run_max_tokens_experiment().items():
        print(f"{label}: {answer}")
    print()

    print("=== stop experiment ===")
    for label, answer in run_stop_experiment().items():
        print(f"{label}: {answer}")
    print()

    print("=== Recommended settings for grounded task ===")
    print(
        "Set temperature near 0.0 to 0.2, max_tokens to a sensible budget (e.g., 120-250), "
        "and optionally use stop to cut off extra verbose output. This favors stable, evidence-based answers "
        "and reduces cost by limiting generation length."
    )


if __name__ == "__main__":
    print_experiment_summary()
