# Structured Output & JSON Response Handling

## Prompt for defined JSON structure

The model is instructed to return only a JSON object with the exact keys `answer` and `source`.

Example system message:

```json
{
  "role": "system",
  "content": "You are a clinical knowledge assistant. Return ONLY valid JSON with exactly these keys: answer, source. The answer must be a string, the source must be a string. Do not include markdown, prose, or code fences."
}
```

## Valid parsed result

```json
{
  "answer": "The refund window is 30 days from purchase with proof of purchase.",
  "source": "hospital_policy_v2.pdf"
}
```

## Malformed JSON example

```json
{"answer": "The refund window is 30 days.", "source": "hospital_policy_v2.pdf" 
```

This is malformed because the closing brace is missing. The application catches the parsing failure, logs the error, and returns a safe fallback instead of crashing.

## Validation rule

Before the result is used, the app checks that:

- both `answer` and `source` exist
- both values are non-empty strings

If either check fails, the response is rejected and handled gracefully.

## Recovery pattern

```python
try:
    parsed = json.loads(raw_response)
    validate_required_fields(parsed)
except ValueError:
    log the error and return a safe fallback
```

This keeps the app stable even when the model emits broken JSON.
