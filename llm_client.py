"""Shared offline-first client for the OpenRouter-compatible mock endpoint."""

import os
import time
import warnings
from typing import Any

import mock_llm


MODEL = "openai/gpt-4o-mini"
URL = mock_llm.URL
BASE_DELAY = 0.05
MAX_429_RETRIES = 3


def call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> str:
    """Return assistant text, retrying only rate-limit responses."""
    if not messages:
        raise ValueError("messages must contain at least one message")
    if max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")

    mode = os.getenv("LLM_MODE", "mock").strip().lower()
    headers = {"Content-Type": "application/json"}
    if mode == "live":
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_MODE=live requires the LLM_API_KEY environment variable")
        import requests

        headers["Authorization"] = f"Bearer {api_key}"
        post = requests.post
    elif mode == "mock":
        headers["Authorization"] = "Bearer mock-token"
        post = mock_llm.post
    else:
        raise ValueError("LLM_MODE must be either 'mock' or 'live'")

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for retry_number in range(MAX_429_RETRIES + 1):
        response = post(URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 429:
            if retry_number == MAX_429_RETRIES:
                raise RuntimeError("LLM request failed with status 429 after 3 retries")
            time.sleep(BASE_DELAY * (2**retry_number))
            continue
        if response.status_code != 200:
            error = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"LLM request failed with status {response.status_code}: {error}"
            )

        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not contain any choices")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("LLM response did not contain assistant text")
        if choice.get("finish_reason") == "length":
            warnings.warn(
                "LLM response was truncated because it reached max_tokens",
                RuntimeWarning,
                stacklevel=2,
            )
        return content

    raise AssertionError("unreachable retry loop")
