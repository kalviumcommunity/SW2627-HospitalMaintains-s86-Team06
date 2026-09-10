# Hallucination guardrail cases

Query: `What medication instructions should the patient follow?`

## Refusal case

Configuration: `minimum_score=1.00`, `minimum_chunks=1`

The retrieved chunks score below `1.00`, so zero sources qualify.

```text
I don't know based on the indexed documents. I could not find enough relevant evidence to answer safely.
```

Pipeline state: `refused=True`  
Reason: `only 0 source(s) reached the 1.00 relevance threshold`

## Confident answer case

Configuration: `minimum_score=0.75`, `minimum_chunks=1`

Retrieved scores: `0.999884`, `0.999676`

```text
Based on the indexed documents, The patient should take the prescribed medication with water. Patients need to use their recommended medicine with water. ([1], [2]).
```

Pipeline state: `refused=False`

The answer is allowed because both medication chunks exceed the relevance
threshold. Its `[1]` and `[2]` markers map to the citation records documented
in `outputs/cited_answers_sample.md`.