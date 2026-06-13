from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import psycopg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register vLLM-served LoRA adapter model handles in Letta."
    )
    parser.add_argument("--base-handle", default="vllm/NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Adapter model name exposed by vLLM, for example nano-memgpt-llama3-r16.",
    )
    parser.add_argument(
        "--pg-uri",
        default=os.getenv(
            "LETTA_PG_URI",
            "postgresql://nanomemgpt:nanomemgpt@pgvector:5432/nanomemgpt",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapters = args.adapter or [
        "nano-memgpt-llama3-r8",
        "nano-memgpt-llama3-r16",
        "nano-memgpt-llama3-query-answer-r16",
        "nano-memgpt-llama3-query-only-r16",
    ]
    with psycopg.connect(args.pg_uri) as connection:
        base = connection.execute(
            """
            SELECT provider_id, organization_id, model_endpoint_type, max_context_window,
                   supports_token_streaming, supports_tool_calling,
                   _created_by_id, _last_updated_by_id
            FROM provider_models
            WHERE handle = %s AND model_type = 'llm' AND is_deleted = FALSE
            """,
            (args.base_handle,),
        ).fetchone()
        if base is None:
            raise SystemExit(f"Base provider model handle not found: {args.base_handle}")

        for adapter in adapters:
            handle = f"vllm/{adapter}"
            model_id = f"model-{uuid.uuid4()}"
            row = connection.execute(
                """
                INSERT INTO provider_models (
                    id, handle, display_name, name, provider_id, organization_id,
                    model_type, enabled, model_endpoint_type, max_context_window,
                    supports_token_streaming, supports_tool_calling,
                    _created_by_id, _last_updated_by_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'llm', TRUE, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name, provider_id, model_type)
                DO UPDATE SET
                    handle = EXCLUDED.handle,
                    display_name = EXCLUDED.display_name,
                    enabled = TRUE,
                    model_endpoint_type = EXCLUDED.model_endpoint_type,
                    max_context_window = EXCLUDED.max_context_window,
                    supports_token_streaming = EXCLUDED.supports_token_streaming,
                    supports_tool_calling = EXCLUDED.supports_tool_calling,
                    updated_at = NOW(),
                    is_deleted = FALSE
                RETURNING id, handle, enabled, is_deleted
                """,
                (
                    model_id,
                    handle,
                    adapter,
                    adapter,
                    base[0],
                    base[1],
                    base[2],
                    base[3],
                    base[4],
                    base[5],
                    base[6],
                    base[7],
                ),
            ).fetchone()
            print(f"registered={row}")


if __name__ == "__main__":
    main()
