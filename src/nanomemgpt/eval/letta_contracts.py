from __future__ import annotations

from copy import deepcopy
from typing import Any

import requests


PAPER_SUBSTRING_DESCRIPTION = """Search prior conversation history using case-insensitive substring matching.

Choose short literal words or phrases that are likely to occur verbatim in the prior
conversation. This local DMR protocol does not use semantic similarity."""

PAPER_SUBSTRING_QUERY_DESCRIPTION = (
    "Short literal word or phrase to match case-insensitively as a substring of prior "
    "conversation messages. Prefer text likely to occur verbatim."
)


def configure_dmr_recall_contract(
    letta_base_url: str,
    mode: str = "paper_substring",
    timeout: float = 60,
) -> dict[str, Any]:
    if mode != "paper_substring":
        raise ValueError(f"Unsupported recall-search contract: {mode}")

    response = requests.get(
        f"{letta_base_url}/v1/tools",
        params={"name": "conversation_search"},
        timeout=timeout,
    )
    response.raise_for_status()
    tools = response.json()
    if len(tools) != 1:
        raise RuntimeError(f"Expected one conversation_search tool, found {len(tools)}.")

    tool = tools[0]
    if mode == "paper_substring":
        schema = deepcopy(tool["json_schema"])
        schema["description"] = PAPER_SUBSTRING_DESCRIPTION
        schema["parameters"]["properties"]["query"]["description"] = (
            PAPER_SUBSTRING_QUERY_DESCRIPTION
        )
        response = requests.patch(
            f"{letta_base_url}/v1/tools/{tool['id']}",
            json={
                "description": PAPER_SUBSTRING_DESCRIPTION,
                "json_schema": schema,
                "metadata_": {
                    **tool.get("metadata_", {}),
                    "nano_memgpt_dmr_recall_contract": mode,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        tool = response.json()

    return {
        "mode": mode,
        "tool_id": tool["id"],
        "description": PAPER_SUBSTRING_DESCRIPTION,
        "query_description": PAPER_SUBSTRING_QUERY_DESCRIPTION,
    }
