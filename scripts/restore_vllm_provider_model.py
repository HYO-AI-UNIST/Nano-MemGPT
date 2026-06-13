from __future__ import annotations

import argparse
import os

import psycopg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a soft-deleted Letta vLLM provider-model record."
    )
    parser.add_argument("--handle", required=True, help="Letta model handle, such as vllm/org/model.")
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
    if not args.handle.startswith("vllm/"):
        raise SystemExit(f"Refusing to modify a non-vLLM handle: {args.handle}")

    with psycopg.connect(args.pg_uri) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE provider_models
                SET is_deleted = FALSE, enabled = TRUE, updated_at = NOW()
                WHERE handle = %s AND model_type = 'llm'
                RETURNING id, handle, enabled, is_deleted
                """,
                (args.handle,),
            )
            restored = cursor.fetchall()

    if len(restored) != 1:
        raise SystemExit(
            f"Expected exactly one provider-model row for {args.handle}, found {len(restored)}."
        )
    print(f"restored={restored[0]}")


if __name__ == "__main__":
    main()
