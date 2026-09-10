# Cited answers and source verification

Query: `What medication instructions should the patient follow?`

## Generated answer

> Based on the indexed documents, The patient should take the prescribed medication with water. Patients need to use their recommended medicine with water. ([1], [2]).

## Citation-to-source mapping

| Marker | Source document | Metadata | Retrieved text | Verification |
| --- | --- | --- | --- | --- |
| `[1]` | `medication-guideline.pdf` | `chunk_index=1`, `section=Dosage` | The patient should take the prescribed medication with water. | `True` |
| `[2]` | `medication-guideline.pdf` | `chunk_index=2`, `section=Administration` | Patients need to use their recommended medicine with water. | `True` |

Each citation is checked against the exact retrieved chunk text. For example,
`verify_citation([1], "The patient should take the prescribed medication with water.")`
returns `True`; checking `[1]` against changed text returns `False`.

## No-source fallback

Query: `What is not covered by the indexed documents?`

> I could not find supporting information in the indexed documents.

This fallback contains no citation because there is no retrieved source to
support one.