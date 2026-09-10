# Hallucination guardrails and refusal handling

Before prompt assembly or answer generation, `evaluate_retrieval` checks the
retrieved evidence. A source qualifies only when its cosine similarity score
is at least the configured `minimum_score` (default `0.75`). The pipeline also
requires `minimum_chunks` qualifying sources (default `1`). Empty retrieval
therefore fails automatically, as does a result set whose scores are too low.

When the check fails, the pipeline returns:

> I don't know based on the indexed documents. I could not find enough relevant evidence to answer safely.

It sets `refused=True` and records the reason, and it does not assemble a
generation prompt. This prevents the model from filling an evidence gap with
an invented answer or citation. Strong sources continue through the existing
grounded, cited generation path. See
[`outputs/guardrail_cases.md`](outputs/guardrail_cases.md) for both cases.

The threshold is intentionally configurable. Raising it reduces unsupported
answers but may refuse more borderline questions; lowering it improves recall
but increases the need for human review.