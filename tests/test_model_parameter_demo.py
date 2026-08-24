from src.model_parameter_experiments import (
    mock_max_tokens_response,
    mock_stop_response,
    mock_temperature_response,
)


def test_temperature_response_is_stable_at_zero():
    response = mock_temperature_response(0.0)
    assert "30 days" in response
    assert "maybe" not in response.lower()


def test_max_tokens_response_is_capped():
    response = mock_max_tokens_response(12)
    words = response.split()
    assert len(words) <= 12


def test_stop_sequence_truncates_response():
    response = mock_stop_response(stop="###")
    assert "###" not in response
    assert "30 days" in response
