import json
import logging
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_client():
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    return OpenAI(base_url=base_url, api_key=api_key)

def run_mock_demo():
    print("\n=======================================================")
    print(" 🚀 RUNNING DEMO IN SIMULATED / OFFLINE MOCK MODE")
    print(" (Use this mode to learn & verify outputs without an API key)")
    print("=======================================================\n")

    # Demo 1
    print("=== Demo 1: System vs User Role Constraints ===")
    system_instruction = "You are a support assistant for an internal docs tool. Answer in 2 sentences max. If you are unsure, say you don't know."
    user_query = "What is our refund window?"
    logging.info("System Instruction: %s", system_instruction)
    logging.info("User Query: %s", user_query)
    mock_reply_1 = "Our standard refund window is 30 days from the purchase date with original proof of purchase. If you need assistance with an order past 30 days, please contact support."
    print(f"\nUser Query: {user_query}")
    print(f"Assistant Response (Constrained to 2 sentences max):\n{mock_reply_1}\n")

    # Demo 2
    print("=== Demo 2: Comparing Prompt Variations (Vague vs. Specific) ===")
    print("Prompt: 'Explain our refund policy.'")
    print(" -> Output: Customers can request a full refund within 30 days of purchase for eligible items in original condition. Shipping fees are non-refundable, and refunds process in 5-7 business days.\n")
    print("Prompt: 'In one sentence, state the refund window in days.'")
    print(" -> Output: The refund window is 30 days from the date of purchase.\n")

    # Demo 3
    print("=== Demo 3: Explicit Format Constraints (JSON Output) ===")
    user_query = "What is the return and refund policy timeframe?"
    print(f"User Query: {user_query}")
    mock_json = json.dumps({"answer": "Full refunds are allowed within 30 days of purchase.", "refund_days": 30, "confidence": "high"}, indent=2)
    print(f"Formatted Response (Strict JSON):\n{mock_json}\n")

def demo_system_vs_user_roles(client, model):
    print("\n=== Demo 1: System vs User Role Constraints ===")
    system_instruction = (
        "You are a support assistant for an internal docs tool. "
        "Answer in 2 sentences max. If you are unsure, say you don't know."
    )
    user_query = "What is our refund window?"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query},
    ]

    logging.info("System Instruction: %s", system_instruction)
    logging.info("User Query: %s", user_query)

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
    content = resp.choices[0].message.content
    print(f"\nUser Query: {user_query}")
    print(f"Assistant Response (Constrained by System Role):\n{content}\n")

def demo_prompt_variations(client, model):
    print("\n=== Demo 2: Comparing Prompt Variations (Vague vs. Specific) ===")
    system_role = "You are concise and factual."
    
    prompts = [
        "Explain our refund policy.",  # Vague prompt
        "In one sentence, state the refund window in days."  # Specific + formatted prompt
    ]

    for p in prompts:
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": p}
        ]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
        content = resp.choices[0].message.content
        print(f"Prompt: '{p}'")
        print(f" -> Output: {content}\n")

def demo_format_constraints(client, model):
    print("\n=== Demo 3: Explicit Format Constraints (JSON Output) ===")
    system_instruction = (
        "You are a helpful customer support bot. "
        "Reply ONLY with a JSON object in the format: {\"answer\": string, \"refund_days\": number, \"confidence\": string}. "
        "Do not include markdown code block formatting or extra commentary."
    )
    user_query = "What is the return and refund policy timeframe?"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query}
    ]

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    content = resp.choices[0].message.content
    print(f"User Query: {user_query}")
    print(f"Formatted Response:\n{content}\n")

def main():
    use_mock = os.getenv("MOCK_MODE", "false").lower() == "true" or "--mock" in sys.argv
    if use_mock:
        run_mock_demo()
        return

    try:
        client = get_client()
        chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        
        demo_system_vs_user_roles(client, chat_model)
        demo_prompt_variations(client, chat_model)
        demo_format_constraints(client, chat_model)

    except AuthenticationError:
        print("\n[401 Unauthorized] API key is missing or invalid.")
        print("Switching automatically to Demo Mock Mode...\n")
        run_mock_demo()
    except Exception as e:
        print(f"\n[API Connection Notice]: {e}")
        print("Switching automatically to Demo Mock Mode...\n")
        run_mock_demo()

if __name__ == "__main__":
    main()
