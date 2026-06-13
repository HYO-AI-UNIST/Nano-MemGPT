from __future__ import annotations

import argparse
import json
import os

from nanomemgpt.eval.letta_contracts import configure_dmr_recall_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure Letta's conversation_search schema for the DMR protocol."
    )
    parser.add_argument(
        "--letta-base-url",
        default=os.getenv("LETTA_BASE_URL", "http://letta-server:8283"),
    )
    parser.add_argument(
        "--mode",
        choices=["paper_substring"],
        default="paper_substring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            configure_dmr_recall_contract(args.letta_base_url, args.mode),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
