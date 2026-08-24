# Model Parameters & Output Control

This note documents a small parameter experiment for a grounded, factual RAG-style answer. The core idea is that generation becomes more stable and lower-cost when the model is constrained with low randomness, a sensible token budget, and optional stop phrases.

## Task 1 — Temperature changes the level of variability

### Prompt used

> You are a factual hospital assistant. Using the hospital guideline, answer: "What is the patient refund window?" Answer in one sentence.

### Example outputs

- Temperature = 0.0
  - The hospital refund window is 30 days from the date of purchase, with proof of purchase required.
- Temperature = 0.4
  - Based on the policy, patients generally have a 30-day refund window from purchase, provided they keep their receipt and the item qualifies.
- Temperature = 1.0
  - The hospital policy is designed to be patient-friendly: in many cases, a customer can request a refund within about 30 days, though some items and services may vary depending on the specific terms and supporting evidence.

### What this shows

- Lower temperatures are more deterministic and grounded.
- Higher temperatures broaden the wording and introduce hedging or softer phrasing.
- For factual RAG answers, low temperature is usually better because it reduces hallucination drift.

## Task 2 — max_tokens caps response length

### Example outputs

- max_tokens = 20
  - The hospital refund window is 30 days from the date of purchase. Proof of purchase is required for a valid refund request.
- max_tokens = 12
  - The hospital refund window is 30 days from the date of purchase.
- max_tokens = 6
  - The hospital refund window is 30 days

### What this shows

- max_tokens controls how many output tokens can be generated.
- Longer values allow a fuller answer; smaller values create concise, budget-aware answers.
- This matters for cost because token usage directly affects model billing.

## Task 3 — stop can cut off a response early

### Example output

- stop = "###"
  - The hospital refund window is 30 days from the date of purchase. Proof of purchase is required.

### What this shows

- stop tells the model when to stop generating text.
- This is useful for cutting off optional sections or keeping answer formatting predictable.
- It can also reduce wasted tokens.

## Task 4 — Recommended settings for a grounded task

Recommended settings for a hospital or RAG answer:

- temperature = 0.0 to 0.2
- max_tokens = 120 to 250 depending on answer depth
- optional stop = a short delimiter or section marker when formatting matters

### Why

- Low temperature makes the answer stable, factual, and consistent with source material.
- max_tokens prevents runaway generation and keeps the answer within budget.
- stop helps enforce answer boundaries and prevents extra, low-value text.

## Summary

For grounded responses, the best default is usually: low temperature, moderate max_tokens, and an optional stop sequence when answer structure matters. That combination improves trust, keeps costs under control, and reduces verbosity.
