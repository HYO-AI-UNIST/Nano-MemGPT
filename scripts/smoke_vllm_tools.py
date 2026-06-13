from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM OpenAI tool calling.")
    parser.add_argument("--model", help="Served model id. Defaults to the first model returned by /v1/models.")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL") or os.getenv("VLLM_HOST_BASE_URL", "http://localhost:8001/v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url
    client = OpenAI(base_url=base_url, api_key=os.getenv("VLLM_API_KEY", "EMPTY"))
    models = client.models.list()
    if not models.data:
        raise SystemExit(f"No models served by {base_url}")
    model_id = args.model or models.data[0].id

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "You manage long-term memory. Use archival_memory_insert whenever the "
                    "user explicitly asks you to remember a durable personal fact."
                ),
            },
            {"role": "user", "content": "Remember that my favorite color is green."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "archival_memory_insert",
                    "description": "Persist a durable fact in archival memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                },
            }
        ],
        tool_choice="auto",
        temperature=0,
    )
    message = response.choices[0].message
    print(f"model={model_id}")
    print(f"content={message.content}")
    print(f"tool_calls={json.dumps([call.model_dump() for call in message.tool_calls or []])}")
    if not message.tool_calls:
        raise SystemExit("Tool-call smoke test failed: model returned no tool call.")


if __name__ == "__main__":
    main()
