import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prompt_templates import render_batch_prompt, render_clinical_prompt


def demo_chat_feature():
    context = "Hospital policy: refunds are allowed within 30 days with proof of purchase."
    question = "What is the refund window for a patient?"
    prompt = render_clinical_prompt(context=context, question=question)
    print("=== Chat feature prompt ===")
    print(prompt)
    print()


def demo_batch_feature():
    context = "Batch record: patient accounts can request reimbursement within 30 days of billing."
    question = "Summarize the reimbursement policy in brief bullet points."
    prompt = render_batch_prompt(context=context, question=question)
    print("=== Batch feature prompt ===")
    print(prompt)
    print()


if __name__ == "__main__":
    demo_chat_feature()
    demo_batch_feature()
