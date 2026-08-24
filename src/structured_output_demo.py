import json
import logging
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REQUIRED_FIELDS = {"answer", "source"}


def prompt_for_structured_json() -> Dict[str, Any]:
    """Return a model prompt that asks for strictly valid JSON."""
    return {
        "role": "system",
        "content": (
            "You are a clinical knowledge assistant. "
            "Return ONLY valid JSON with exactly these keys: answer, source. "
            "The answer must be a string, the source must be a string. "
            "Do not include markdown, prose, or code fences."
        ),
    }


def parse_json_response(raw_response: str) -> Dict[str, Any]:
    """Parse a response body and return a dict or raise a ValueError."""
    if raw_response is None:
        raise ValueError("No response content received from model.")

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")

    return parsed


def validate_required_fields(payload: Dict[str, Any]) -> None:
    """Ensure the payload contains all required keys and valid values."""
    missing = sorted(REQUIRED_FIELDS - set(payload.keys()))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if not isinstance(payload.get("answer"), str) or not payload["answer"].strip():
        raise ValueError("Field 'answer' must be a non-empty string.")

    if not isinstance(payload.get("source"), str) or not payload["source"].strip():
        raise ValueError("Field 'source' must be a non-empty string.")


def safe_parse_response(raw_response: str) -> Optional[Dict[str, Any]]:
    """Parse and validate a model response safely, returning None on failure."""
    try:
        parsed = parse_json_response(raw_response)
        validate_required_fields(parsed)
        return parsed
    except ValueError as exc:
        logging.error("JSON validation failed: %s", exc)
        return None


def demo_structured_output() -> None:
    sample_good = '{"answer":"The refund window is 30 days from purchase with proof of purchase.","source":"hospital_policy_v2.pdf"}'
    malformed_sample = '{"answer": "The refund window is 30 days.", "source": "hospital_policy_v2.pdf"'

    good_result = safe_parse_response(sample_good)
    malformed_result = safe_parse_response(malformed_sample)

    print("=== Structured JSON prompt ===")
    print(json.dumps(prompt_for_structured_json(), indent=2))
    print()

    print("=== Valid JSON parse ===")
    print(json.dumps(good_result, indent=2))
    print()

    print("=== Malformed JSON recovery ===")
    print(f"Recovered result: {malformed_result}")
    print("Failure handled gracefully without crashing.")


if __name__ == "__main__":
    demo_structured_output()
