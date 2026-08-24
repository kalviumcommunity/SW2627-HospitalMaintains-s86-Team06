from string import Template


BASE_PROMPT_TEMPLATE = Template(
    "You are a $role. Use the following context to answer the question. "
    "Answer in $style.\n\nContext:\n$context\n\nQuestion:\n$question"
)


def render_prompt(role: str, context: str, question: str, style: str = "concise factual sentences") -> str:
    return BASE_PROMPT_TEMPLATE.substitute(
        role=role,
        context=context,
        question=question,
        style=style,
    )


def render_clinical_prompt(context: str, question: str) -> str:
    return render_prompt(
        role="clinical knowledge assistant",
        context=context,
        question=question,
        style="brief, evidence-based sentences",
    )


def render_batch_prompt(context: str, question: str) -> str:
    return render_prompt(
        role="batch processing assistant",
        context=context,
        question=question,
        style="short, structured bullet points",
    )
