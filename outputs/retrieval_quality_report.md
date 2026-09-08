# Retrieval Quality Sanity Report

Test count: 3
Passes: 3
Failures: 0

## Top-ranked sources

- medication-guidance: The patient should take the prescribed medication with water. (score=0.999884, source=medication-guideline.pdf)
- medication-paraphrase: Patients need to use their recommended medicine with water. (score=1.000000, source=medication-guideline.pdf)
- account-access: The help desk can reset an employee password. (score=1.000000, source=it-support-handbook.pdf)

## Result details

- medication-guidance: related_above_unrelated=True, expected_top_ranked=True, relevant_score=0.999884, unrelated_score=0.076703, score_gap=0.000208
  Note: The dosage guidance should outrank the unrelated IT account-access content.
- medication-paraphrase: related_above_unrelated=True, expected_top_ranked=True, relevant_score=1.000000, unrelated_score=0.089437, score_gap=0.000579
  Note: This is a borderline paraphrase case: both medication chunks are very similar, so the top result should be close but still preferred.
- account-access: related_above_unrelated=True, expected_top_ranked=True, relevant_score=1.000000, unrelated_score=0.076448, score_gap=0.910563
  Note: An IT support query should prefer the account-access chunk over clinical guidance.

## Borderline or surprising case

- medication-guidance: the top result is only 0.000208 above the next rank, which indicates a borderline semantic match. This revealed a near-duplicate medication pair in the corpus, so the metric is still behaving sensibly but the corpus contains close paraphrases that can produce narrow score gaps.
- medication-paraphrase: the top result is only 0.000579 above the next rank, which indicates a borderline semantic match. This revealed a near-duplicate medication pair in the corpus, so the metric is still behaving sensibly but the corpus contains close paraphrases that can produce narrow score gaps.
