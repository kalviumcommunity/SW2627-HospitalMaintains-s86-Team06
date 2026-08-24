from src.structured_output_demo import safe_parse_response, validate_required_fields


def test_valid_json_parses_and_validates():
    payload = '{"answer": "The refund window is 30 days.", "source": "policy_v2.pdf"}'
    result = safe_parse_response(payload)
    assert result == {"answer": "The refund window is 30 days.", "source": "policy_v2.pdf"}


def test_missing_field_is_rejected():
    payload = '{"answer": "The refund window is 30 days."}'
    result = safe_parse_response(payload)
    assert result is None


def test_malformed_json_is_recovered_gracefully():
    invalid = '{"answer": "The refund window is 30 days.", "source": "policy_v2.pdf"'
    result = safe_parse_response(invalid)
    assert result is None


def test_validate_required_fields_raises_for_wrong_type():
    try:
        validate_required_fields({"answer": 123, "source": "policy_v2.pdf"})
        assert False, "Expected ValueError to be raised"
    except ValueError:
        pass
