# Augmented prompt sample

Query: `What medication instructions should the patient follow?`

Configuration:

- Model token budget: `256`
- Reserved answer tokens: `64`
- Included sources: `2`
- Prompt tokens: `124`
- Prompt plus reserved answer: `188`
- Budget remaining: `68`

The complete augmented prompt sent to a chat model is:

### System message

> Answer only from the provided context. Cite supporting passages with their source marker, such as [1]. If the context is insufficient, say that you do not have enough information and do not guess.

### User message

```text
Context:
[1] medication-guideline.pdf | section: Dosage
The patient should take the prescribed medication with water.

[2] medication-guideline.pdf | section: Administration
Patients need to use their recommended medicine with water.

Question: What medication instructions should the patient follow?
```

Both retrieved chunks fit within the budget, and the `[1]` and `[2]` markers
allow the generated answer to reference the returned source evidence.