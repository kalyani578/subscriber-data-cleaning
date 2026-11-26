"""Run the four offline demonstrations for the Part D LLM client."""

import os
import warnings

import mock_llm
from llm_client import call_llm


def run_demos() -> None:
    """Verify authentication errors, truncation, conversation memory, and retries."""
    os.environ["LLM_MODE"] = "mock"
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Please introduce yourself."}],
        "temperature": 0.0,
        "max_tokens": 300,
    }

    mock_llm.reset()
    no_auth = mock_llm.post(mock_llm.URL, headers={}, json=payload, timeout=30)
    wrong_url = mock_llm.post(
        "https://openrouter.ai/wrong",
        headers={"Authorization": "Bearer mock-token"},
        json=payload,
        timeout=30,
    )
    assert no_auth.status_code == 401
    assert wrong_url.status_code == 404
    print(f"missing Authorization: {no_auth.status_code}")
    print(f"wrong URL: {wrong_url.status_code}")

    mock_llm.reset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        short_text = call_llm(payload["messages"], max_tokens=3)
    assert any("truncated" in str(item.message) for item in caught)
    print(f"max_tokens=3: finish_reason=length; returned text={short_text!r}")

    mock_llm.reset()
    remembered = call_llm(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Asha."},
            {"role": "assistant", "content": "Hello, Asha."},
            {"role": "user", "content": "What is my name?"},
        ]
    )
    assert remembered == "Your name is Asha."
    print(f"conversation memory: {remembered}")

    mock_llm.reset()
    for _ in range(30):
        call_llm([{"role": "user", "content": "ping"}])
    total_calls = mock_llm._state["calls"]
    recovered_rate_limits = total_calls - 30
    assert total_calls == 33
    assert recovered_rate_limits == 3
    print(
        "30 successful requests; "
        f"{recovered_rate_limits} rate limits recovered; {total_calls} total mock calls"
    )


if __name__ == "__main__":
    run_demos()
