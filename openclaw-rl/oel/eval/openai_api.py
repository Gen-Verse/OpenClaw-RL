"""OpenAI API client wrapper for evaluation.

Configure via environment variables:
    OPENAI_API_KEY      - API key (required)
    OPENAI_BASE_URL     - Base URL (default: https://api.openai.com/v1)
    OPENAI_MODEL        - Model name (default: gpt-4.1)

Example:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_MODEL="gpt-4.1"
"""

import os
from openai import OpenAI


def get_client(model_name: str = None, **kwargs):
    """Return (client, model_name) tuple compatible with OpenAI API.

    Args:
        model_name: Model to use. Falls back to OPENAI_MODEL env var, then "gpt-4.1".
        **kwargs: Additional keyword arguments (ignored, for backward compatibility).

    Returns:
        Tuple of (OpenAI client, resolved model name).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", None)

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it to your OpenAI API key."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=kwargs.get("max_retries", 5),
    )

    resolved_model = model_name or os.environ.get("OPENAI_MODEL", "gpt-4.1")
    return client, resolved_model


if __name__ == "__main__":
    client, model = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=100,
    )
    print(f"Model: {model}")
    print(f"Response: {response.choices[0].message.content}")
