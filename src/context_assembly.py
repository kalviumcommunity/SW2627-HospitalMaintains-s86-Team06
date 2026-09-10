"""Budget-aware context injection for grounded RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.token_counter import count_tokens


GROUNDING_INSTRUCTIONS = (
    "Answer only from the provided context. Cite supporting passages with their "
    "source marker, such as [1]. If the context is insufficient, say that you "
    "do not have enough information and do not guess."
)


@dataclass(frozen=True)
class AugmentedPrompt:
    """Prompt messages and accounting details for one grounded model call."""

    system_message: str
    user_message: str
    context: str
    included_sources: int
    prompt_tokens: int
    reserved_answer_tokens: int
    model_token_budget: int

    @property
    def total_reserved_tokens(self) -> int:
        """Return prompt tokens plus the answer allowance."""
        return self.prompt_tokens + self.reserved_answer_tokens


def _source_block(index: int, source: Any) -> str:
    filename = source.metadata.get("source_document", "unknown source")
    section = source.metadata.get("section")
    location = f" | section: {section}" if section else ""
    return f"[{index}] {filename}{location}\n{source.text}"


def build_augmented_prompt(
    query: str,
    sources: Sequence[Any],
    model_token_budget: int = 256,
    reserved_answer_tokens: int = 64,
) -> AugmentedPrompt:
    """Inject as many ranked sources as fit while preserving answer capacity.

    Sources are considered in retrieval order. A source that does not fit is
    skipped, allowing smaller later chunks to remain useful. The returned
    accounting guarantees prompt tokens plus the reserved answer allowance do
    not exceed the model budget.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if model_token_budget <= 0 or reserved_answer_tokens < 0:
        raise ValueError("token budgets must be positive and non-negative")
    if reserved_answer_tokens >= model_token_budget:
        raise ValueError("reserved answer tokens must leave room for the prompt")

    system_message = GROUNDING_INSTRUCTIONS
    selected_blocks: list[str] = []

    def make_user_message() -> str:
        context = "\n\n".join(selected_blocks) or "[No supporting context was retrieved.]"
        return f"Context:\n{context}\n\nQuestion: {query}"

    for source_index, source in enumerate(sources, start=1):
        selected_blocks.append(_source_block(source_index, source))
        candidate_user_message = make_user_message()
        candidate_tokens = count_tokens(system_message) + count_tokens(candidate_user_message)
        if candidate_tokens + reserved_answer_tokens > model_token_budget:
            selected_blocks.pop()

    user_message = make_user_message()
    prompt_tokens = count_tokens(system_message) + count_tokens(user_message)
    if prompt_tokens + reserved_answer_tokens > model_token_budget:
        raise ValueError(
            "model token budget is too small for grounding instructions, "
            "the question, and the reserved answer"
        )
    return AugmentedPrompt(
        system_message=system_message,
        user_message=user_message,
        context="\n\n".join(selected_blocks),
        included_sources=len(selected_blocks),
        prompt_tokens=prompt_tokens,
        reserved_answer_tokens=reserved_answer_tokens,
        model_token_budget=model_token_budget,
    )
