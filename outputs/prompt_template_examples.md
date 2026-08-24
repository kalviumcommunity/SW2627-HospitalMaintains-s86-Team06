# Prompt Templates & Reusable Prompt Design

This project keeps prompt text separate from runtime logic by storing templates in the template module and rendering them with values at runtime.

## Template definition

```python
BASE_PROMPT_TEMPLATE = Template(
    "You are a $role. Use the following context to answer the question. "
    "Answer in $style.\n\nContext:\n$context\n\nQuestion:\n$question"
)
```

## Runtime injection

```python
prompt = render_clinical_prompt(
    context="Hospital policy: refunds are allowed within 30 days with proof of purchase.",
    question="What is the refund window for a patient?",
)
```

## Example renders

### Chat feature render

```text
You are a clinical knowledge assistant. Use the following context to answer the question. Answer in brief, evidence-based sentences.

Context:
Hospital policy: refunds are allowed within 30 days with proof of purchase.

Question:
What is the refund window for a patient?
```

### Batch feature render

```text
You are a batch processing assistant. Use the following context to answer the question. Answer in short, structured bullet points.

Context:
Batch record: patient accounts can request reimbursement within 30 days of billing.

Question:
Summarize the reimbursement policy in brief bullet points.
```

## Why this helps

- One template can power multiple features.
- Prompt text is centralized and easier to version or update.
- Business logic does not need to be edited when the wording changes.
- Reuse improves consistency across chat, CLI, and batch tasks.
