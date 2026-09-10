import pytest

from src.context_assembly import GROUNDING_INSTRUCTIONS, build_augmented_prompt
from src.rag_pipeline import RetrievedSource


def test_augmented_prompt_injects_markers_filenames_and_grounding_rules():
    sources = [
        RetrievedSource("Take with water.", {"source_document": "dosage.pdf", "section": "Dosage"}, 0.9),
        RetrievedSource("Use the prescribed medicine.", {"source_document": "admin.pdf"}, 0.8),
    ]

    prompt = build_augmented_prompt("How should it be taken?", sources)

    assert "[1] dosage.pdf | section: Dosage" in prompt.context
    assert "[2] admin.pdf" in prompt.context
    assert "Answer only from the provided context" in prompt.system_message
    assert "If the context is insufficient" in prompt.system_message
    assert prompt.user_message.endswith("Question: How should it be taken?")


def test_augmented_prompt_reserves_answer_budget_and_drops_oversized_context():
    huge_source = RetrievedSource("word " * 200, {"source_document": "huge.pdf"}, 0.9)

    prompt = build_augmented_prompt(
        "What does the policy say?",
        [huge_source],
        model_token_budget=100,
        reserved_answer_tokens=20,
    )

    assert prompt.included_sources == 0
    assert prompt.context == ""
    assert prompt.total_reserved_tokens <= prompt.model_token_budget
    assert "[No supporting context was retrieved.]" in prompt.user_message


def test_impossible_budget_is_rejected_clearly():
    with pytest.raises(ValueError, match="too small"):
        build_augmented_prompt(
            "What does the policy say?",
            [],
            model_token_budget=80,
            reserved_answer_tokens=20,
        )


def test_small_sources_fit_and_accounting_is_explicit():
    source = RetrievedSource("Use water.", {"source_document": "guide.pdf"}, 0.9)

    prompt = build_augmented_prompt(
        "What should the patient use?",
        [source],
        model_token_budget=100,
        reserved_answer_tokens=25,
    )

    assert prompt.included_sources == 1
    assert prompt.prompt_tokens + prompt.reserved_answer_tokens <= 100
    assert prompt.system_message == GROUNDING_INSTRUCTIONS