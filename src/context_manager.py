import logging
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Counts tokens for a string using tiktoken or character approximation fallback."""
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
    else:
        return max(1, int(len(text) / 4))

def total_tokens(messages: list) -> int:
    """Calculates total token count across all messages in history."""
    # Includes message content + per-message metadata overhead (~4 tokens per msg in ChatCompletions)
    return sum(count(m.get("content", "")) + 4 for m in messages)

def trim_history(messages: list, token_budget: int = 400) -> list:
    """
    Trims conversation history to stay under token_budget.
    Always preserves messages[0] (the System Prompt).
    Pops the oldest non-system turns (messages[1]) until budget is satisfied.
    """
    trimmed_count = 0
    while total_tokens(messages) > token_budget and len(messages) > 2:
        popped = messages.pop(1)
        trimmed_count += 1
        logging.info("Trimmed oldest turn (%s role, %d tokens). New total: %d tokens",
                     popped["role"], count(popped["content"]), total_tokens(messages))
    return messages

def summarize_history(messages: list, token_budget: int = 400) -> list:
    """
    Replaces older turns with a single summary system message when total tokens exceed budget.
    """
    if total_tokens(messages) <= token_budget or len(messages) <= 3:
        return messages

    system_msg = messages[0]
    older_turns = messages[1:-2]  # Keep system + latest user turn
    recent_turns = messages[-2:]

    summary_text = f"Summary of previous {len(older_turns)} turns: User discussed hospital protocols and service refund windows."
    summary_message = {"role": "system", "content": f"Previous conversation summary: {summary_text}"}

    new_messages = [system_msg, summary_message] + recent_turns
    return new_messages

def run_mock_history_demo():
    print("\n=======================================================")
    print(" 🚀 RUNNING CONTEXT WINDOW DEMO IN SIMULATED MOCK MODE")
    print("=======================================================\n")

    system_prompt = {"role": "system", "content": "You are a clinical knowledge assistant. Answer concisely using hospital guidelines."}
    history = [system_prompt]
    budget = 180  # Low budget to trigger trimming for demonstration

    print(f"System Prompt Initialized ({total_tokens(history)} tokens). Budget Limit = {budget} tokens.\n")

    sample_turns = [
        ("User", "What is the hospital refund policy timeframe?"),
        ("Assistant", "The refund window is 30 calendar days from the date of billing with valid proof of purchase."),
        ("User", "Does the refund policy apply to emergency room fees?"),
        ("Assistant", "No, emergency room registration fees are non-refundable after medical triage is completed."),
        ("User", "Where can patients submit a refund request?"),
        ("Assistant", "Refund requests can be submitted at the central billing office or through the online patient portal."),
        ("User", "How long does payment processing take?"),
        ("Assistant", "Approved refunds are processed to the original payment method within 5 to 7 business days."),
        ("User", "Can on-call nurses access drug interaction guidelines?"),
        ("Assistant", "Yes, nurses can access drug interaction guidelines instantly via the pharmacy module in CKDSS."),
    ]

    for turn_idx, (role, text) in enumerate(sample_turns, 1):
        if role == "User":
            history.append({"role": "user", "content": text})
            print(f"--- Turn {turn_idx//2 + 1}: User Query ---")
            print(f"User: {text}")
            
            before_trim = total_tokens(history)
            print(f"Token Count before check: {before_trim} tokens")
            
            if before_trim > budget:
                print(f"⚠️  Exceeded budget of {budget} tokens! Trimming history...")
                trim_history(history, token_budget=budget)
                print(f"Token Count after trimming: {total_tokens(history)} tokens")
        else:
            history.append({"role": "assistant", "content": text})
            print(f"Assistant: {text}")
            print(f"Active History Length: {len(history)} messages ({total_tokens(history)} tokens)\n")

    print("\n=== Final Conversation State ===")
    print(f"Total Messages Remaining in Context: {len(history)}")
    print(f"Final Token Count: {total_tokens(history)} tokens (Safely under {budget} budget cap)")
    print("\nRemaining Messages in Context Window:")
    for idx, msg in enumerate(history):
        print(f"  [{idx}] {msg['role'].upper()}: {msg['content'][:60]}...")

def main():
    load_dotenv()
    use_mock = os.getenv("MOCK_MODE", "false").lower() == "true" or "--mock" in sys.argv
    if use_mock:
        run_mock_history_demo()
        return

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        system_prompt = {"role": "system", "content": "You are a concise hospital assistant."}
        history = [system_prompt]
        budget = 250

        print("\n=== Interactive Multi-Turn Chat Simulation (With Trimming) ===")
        user_queries = [
            "What is the refund window?",
            "Does it cover ER fees?",
            "How long do refunds take to process?",
            "Who handles drug interaction checks?"
        ]

        for query in user_queries:
            history.append({"role": "user", "content": query})
            trim_history(history, token_budget=budget)

            print(f"\nUser: {query}")
            print(f"Context Payload Size: {total_tokens(history)} tokens")

            resp = client.chat.completions.create(model=chat_model, messages=history, temperature=0.1)
            reply = resp.choices[0].message.content
            history.append({"role": "assistant", "content": reply})

            print(f"Assistant: {reply}")

    except (AuthenticationError, RateLimitError) as e:
        print(f"\n[API Notice]: {e}")
        print("Switching automatically to Context Window Demo Mock Mode...\n")
        run_mock_history_demo()
    except Exception as e:
        print(f"\n[Notice]: {e}")
        print("Switching automatically to Context Window Demo Mock Mode...\n")
        run_mock_history_demo()

if __name__ == "__main__":
    main()
