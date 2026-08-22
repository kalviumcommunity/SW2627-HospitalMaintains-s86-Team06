import os
import sys

# Try importing tiktoken, or provide a fallback tokenizer approximation
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Counts tokens for a given text string.
    Uses tiktoken if installed; otherwise falls back to a character/word token approximation.
    """
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
    else:
        # Fallback estimation: ~4 chars per token in English
        return max(1, int(len(text) / 4))

def estimate_call_cost(prompt_text: str, answer_text: str, input_rate_per_1k: float = 0.0005, output_rate_per_1k: float = 0.0015) -> dict:
    """
    Calculates estimated API cost based on token counts for input (prompt) and output (answer).
    """
    in_tok = count_tokens(prompt_text)
    out_tok = count_tokens(answer_text)
    
    in_cost = (in_tok / 1000.0) * input_rate_per_1k
    out_cost = (out_tok / 1000.0) * output_rate_per_1k
    total_cost = in_cost + out_cost
    
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "input_cost": in_cost,
        "output_cost": out_cost,
        "total_cost": total_cost
    }

def main():
    print("\n=======================================================")
    print(f" 🧮 TOKENIZATION & COST ESTIMATION DEMO (tiktoken: {'Available' if HAS_TIKTOKEN else 'Fallback Approximation'})")
    print("=======================================================\n")

    # 1. Compare text length vs word count vs token count
    print("--- 1. Token Count vs Word Count vs Character Count ---")
    sample_texts = [
        "refund",
        "refundable",
        "What is our refund window?",
    ]

    policy_file_path = os.path.join("data", "policy.txt")
    if os.path.exists(policy_file_path):
        with open(policy_file_path, "r", encoding="utf-8") as f:
            sample_texts.append(f.read())
    else:
        sample_texts.append("Sample hospital clinical policy document content for token counting.")

    for s in sample_texts:
        snippet = s[:45].replace("\n", " ") + ("..." if len(s) > 45 else "")
        chars = len(s)
        words = len(s.split())
        tokens = count_tokens(s)
        print(f"Text: '{snippet}'")
        print(f"  -> Characters: {chars} | Words: {words} | Tokens: {tokens}\n")

    # 2. Estimate cost for a typical RAG call
    print("--- 2. Single Call Cost Estimation ---")
    sample_prompt = (
        "System: You are a clinical assistant. Use the following context to answer.\n"
        "Context: Hospital refund window is 30 calendar days for billing services.\n"
        "User: What is our refund window?"
    )
    sample_answer = "The refund window for hospital billing services is 30 calendar days."

    cost_info = estimate_call_cost(sample_prompt, sample_answer)
    print(f"Input Tokens (Prompt): {cost_info['input_tokens']}")
    print(f"Output Tokens (Answer): {cost_info['output_tokens']}")
    print(f"Total Tokens: {cost_info['total_tokens']}")
    print(f"Estimated Cost: ${cost_info['total_cost']:.6f}\n")

    # 3. Context Window & Corpus Scale Analysis
    print("--- 3. Corpus Scale & Context Window Estimate ---")
    doc_count = 4000
    avg_tokens_per_doc = 500
    total_corpus_tokens = doc_count * avg_tokens_per_doc
    
    # Embedding cost estimation (e.g. text-embedding-3-large at ~$0.00013 per 1k tokens)
    embedding_cost = (total_corpus_tokens / 1000.0) * 0.00013

    print(f"Corpus Size: {doc_count:,} documents (~{avg_tokens_per_doc} tokens/doc)")
    print(f"Total Corpus Tokens: {total_corpus_tokens:,} tokens")
    print(f"Estimated One-Time Ingestion Embedding Cost: ${embedding_cost:.4f}")

    # RAG Query cost scaling (e.g. retrieving top-5 chunks of 300 tokens each per query)
    chunks_per_query = 5
    tokens_per_chunk = 300
    query_context_tokens = chunks_per_query * tokens_per_chunk
    rag_prompt_tokens = query_context_tokens + 100  # 100 system/query tokens
    rag_answer_tokens = 150
    
    single_rag_cost = estimate_call_cost("a" * (rag_prompt_tokens * 4), "a" * (rag_answer_tokens * 4))["total_cost"]
    queries_per_month = 10000
    monthly_rag_cost = single_rag_cost * queries_per_month

    print(f"Retrieved Context per Query: {query_context_tokens} tokens ({chunks_per_query} chunks)")
    print(f"Cost per Single RAG Query: ${single_rag_cost:.5f}")
    print(f"Estimated Monthly Cost (10,000 queries): ${monthly_rag_cost:.2f}\n")

if __name__ == "__main__":
    main()
