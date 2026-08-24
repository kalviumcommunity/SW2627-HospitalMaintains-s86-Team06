from src.prompt_templates import render_batch_prompt, render_clinical_prompt, render_prompt


def test_render_prompt_substitutes_values():
    prompt = render_prompt(
        role="doctor assistant",
        context="Refunds are valid for 30 days.",
        question="What is the refund policy?",
        style="one sentence",
    )
    assert "doctor assistant" in prompt
    assert "Refunds are valid for 30 days." in prompt
    assert "What is the refund policy?" in prompt


def test_clinical_prompt_is_reusable():
    prompt = render_clinical_prompt(
        context="Policy: 30-day refund window.",
        question="How long is the refund window?",
    )
    assert "clinical knowledge assistant" in prompt
    assert "Policy: 30-day refund window." in prompt


def test_batch_prompt_is_reusable():
    prompt = render_batch_prompt(
        context="Batch note: 30-day reimbursement period.",
        question="List the reimbursement details.",
    )
    assert "batch processing assistant" in prompt
    assert "List the reimbursement details." in prompt
