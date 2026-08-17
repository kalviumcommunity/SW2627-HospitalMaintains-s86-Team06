import os

from dotenv import load_dotenv
from openai import OpenAI
import chromadb


def main() -> None:
    load_dotenv()

    api_base_url = os.getenv("API_BASE_URL", "not set")
    api_key = os.getenv("API_KEY", "not set")
    chat_model = os.getenv("CHAT_MODEL", "not set")
    embedding_model = os.getenv("EMBEDDING_MODEL", "not set")

    print("Health check passed: dependencies imported and environment loaded.")
    print(f"API_BASE_URL={api_base_url}")
    print(f"API_KEY={'set' if api_key and api_key != 'not set' else 'not set'}")
    print(f"CHAT_MODEL={chat_model}")
    print(f"EMBEDDING_MODEL={embedding_model}")


if __name__ == "__main__":
    main()
