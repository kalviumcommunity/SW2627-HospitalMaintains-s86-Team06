# Source citation and attribution

The pipeline creates a `SourceCitation` for every retrieved chunk used in an
answer. The citation marker is assigned in retrieval order, while the record
retains the source document, chunk metadata, similarity score, and exact
retrieved text. This makes `[1]` and `[2]` verifiable rather than decorative.

`verify_citation` compares the citation's stored text with the original
retrieved chunk. A changed or missing chunk fails verification. The answer
generator also returns an uncited fallback when there are no supporting
sources, so it cannot invent a citation for an unsupported answer.

The generated sample answer, mappings, verification results, and fallback are
in [`outputs/cited_answers_sample.md`](outputs/cited_answers_sample.md).

## Flow

```text
retrieved chunk -> SourceCitation([N], metadata, exact text)
                 -> grounded answer references [N]
                 -> verifier checks [N] against original retrieved text
```