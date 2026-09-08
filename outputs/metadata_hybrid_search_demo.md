# Metadata Filtering & Hybrid Search Report

Query: **"What are the password reset steps?"**
Metadata Filter Applied: `{'section': 'Account access'}`
Hybrid Keywords: `['password', 'reset']`

## Executive Summary

- **Metadata Filtering**: Restricts retrieval scope before/during vector search to records matching specific metadata criteria (e.g. section, source, department, date, access level).
- **Semantic Vector Search vs Lexical Keyword Search**:
  - *Vector Search (Semantic)*: Matches meaning even when exact words differ using high-dimensional embeddings.
  - *Keyword Search (Lexical)*: Rewards exact word, name, product ID, or code occurrences.
  - *Hybrid Search*: Combines semantic similarity scores with lexical match weights to optimize overall precision.

## Unfiltered Vector Retrieval

Unfiltered queries search the entire corpus. While semantic vector search surfaces high-similarity chunks, plausible-sounding chunks from unrelated sections (e.g. general security policy or campus IT) may dilute precision.

1. **Score:** `0.9999` | **Source:** `medication-guideline.pdf` | **Section:** `Dosage`
   **Text:** The patient should take the prescribed medication with water following post-surgery clinical guidelines.

2. **Score:** `0.9974` | **Source:** `triage-protocol.pdf` | **Section:** `Emergency Care`
   **Text:** Emergency triage procedures require staff to verify patient identity before administering any clinical care.

3. **Score:** `0.1433` | **Source:** `campus-it-guide.pdf` | **Section:** `Campus IT`
   **Text:** Campus IT infrastructure maintains wireless routers and account provisioning for campus facilities.

## Metadata-Filtered Retrieval

Filtering by `{'section': 'Account access'}` ensures that retrieval targets only chunks explicitly belonging to the intended scope. Unrelated sections are pruned out, dramatically raising precision.

1. **Score:** `0.0880` | **Source:** `it-support-handbook.pdf` | **Section:** `Account access`
   **Text:** To reset your employee password, navigate to the self-service IT help desk portal and follow the password reset prompt.

## Hybrid Lexical-Semantic Search

Hybrid search combines vector scores (`vector_weight=0.8`) with keyword match counts (`keyword_weight=0.2`). This preserves semantic ranking while guaranteeing top priority for exact term matches.

1. **Hybrid Score:** `0.4704` (Vector: `0.0880`, Keyword Matches: `2`)
   **Source:** `it-support-handbook.pdf` | **Section:** `Account access`
   **Text:** To reset your employee password, navigate to the self-service IT help desk portal and follow the password reset prompt.

## Trade-offs & Strategic Guidance

1. **Precision vs Recall**:
   - Metadata filtering **improves precision** by eliminating plausible but wrong areas of the corpus.
   - Overly strict filters or missing metadata tags can **hurt recall** by excluding relevant passages.
2. **When to Use Hybrid Search**:
   - Highly recommended when users query exact product names, policy IDs, error codes, course codes, or names where exact-match precision is essential.
