import logging
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_mock_chat():
    print("\n=======================================================")
    print(" 🚀 RUNNING CHAT COMPLETION IN OFFLINE DEMO MODE")
    print("=======================================================\n")

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]

    logging.info("REQUEST: %s", messages)
    mock_reply = "Hello! I am your AI assistant, ready to help you build reliable RAG applications."
    mock_usage = {"prompt_tokens": 18, "completion_tokens": 16, "total_tokens": 34}

    logging.info("RESPONSE: %s", mock_reply)
    logging.info("USAGE: %s", mock_usage)

    print("\n--- Assistant Reply ---")
    print(mock_reply)
    print("-----------------------\n")

def run_chat_completion():
    load_dotenv()
    use_mock = os.getenv("MOCK_MODE", "false").lower() == "true" or "--mock" in sys.argv
    if use_mock:
        run_mock_chat()
        return

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]

    logging.info("REQUEST: %s", messages)

    try:
        resp = client.chat.completions.create(
            model=chat_model,
            messages=messages,
        )

        content = resp.choices[0].message.content
        logging.info("RESPONSE: %s", content)
        logging.info("USAGE: %s", resp.usage)

        print("\n--- Assistant Reply ---")
        print(content)
        print("-----------------------\n")
        return resp

    except AuthenticationError as e:
        logging.error("Auth failed (401): check API_KEY / OPENAI_API_KEY in your .env")
        print(f"\n[401 Unauthorized] {e}")
        print("Switching automatically to Demo Mock Mode...\n")
        run_mock_chat()
    except RateLimitError as e:
        logging.error("Rate limited (429): slow down and retry with backoff")
        print(f"\n[429 Too Many Requests] {e}")
        print("Switching automatically to Demo Mock Mode...\n")
        run_mock_chat()
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
        print("Switching automatically to Demo Mock Mode...\n")
        run_mock_chat()

if __name__ == "__main__":
    run_chat_completion()
