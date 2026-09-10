# Context injection and prompt augmentation

The query-time RAG flow retrieves ranked chunks, then
`build_augmented_prompt` injects those chunks into the user message. Every
chunk receives a stable marker (`[1]`, `[2]`, ...) and its source filename so a
model can cite the evidence it used.

The builder counts the system instructions and user message with the repo's
`count_tokens` helper. It adds sources in retrieval order only while

```text
prompt tokens + reserved answer tokens <= model token budget
```

This preserves room for the answer. A later chunk that is too large is skipped;
the prompt can still use smaller chunks that follow it. If the budget cannot
fit even the grounding instructions, question, and answer reservation, the
builder raises a clear `ValueError` because no valid prompt can be produced.

The system instruction tells the model to answer only from the provided
context, cite source markers, and say when the context is insufficient. This
prevents unsupported claims from being presented as retrieved evidence.

## Sample

The exact generated prompt and token accounting are in
[`outputs/augmented_prompt_sample.md`](outputs/augmented_prompt_sample.md).
Run the sample with:

```powershell
python -c "from src.rag_pipeline import run_rag_pipeline; from src.context_assembly import build_augmented_prompt; r=run_rag_pipeline(); p=build_augmented_prompt(r.query, r.sources); print(p.user_message)"
```