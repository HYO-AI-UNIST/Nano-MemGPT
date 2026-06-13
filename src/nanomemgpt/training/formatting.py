from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from nanomemgpt.trajectory.schema import TrajectoryStep


def format_function_call_target(step: TrajectoryStep) -> str:
    if step.teacher_action is None:
        return step.target_text or ""
    payload = {
        "name": step.teacher_action.name,
        "arguments": step.teacher_action.arguments,
        "request_heartbeat": step.teacher_action.request_heartbeat,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _explicit_tool_call_content(tool_calls: list[dict[str, Any]]) -> str:
    calls = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": raw_arguments}
        else:
            arguments = raw_arguments
        calls.append(
            json.dumps(
                {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(calls)


def normalize_chat_context(context: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Render structured tool-call history as text accepted by basic chat templates."""

    messages = []
    for message in context:
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []
        if content is None and tool_calls:
            content = _explicit_tool_call_content(tool_calls)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        messages.append({"role": str(message.get("role", "user")), "content": content})
    return messages


def render_chat_prompt(step: TrajectoryStep, tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        normalize_chat_context(step.context),
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_chat_sft_records(
    steps: Iterable[TrajectoryStep],
    tokenizer: Any,
) -> Iterable[dict[str, str]]:
    eos_token = tokenizer.eos_token or ""
    for step in steps:
        yield {
            "sample_id": step.sample_id,
            "prompt": render_chat_prompt(step, tokenizer),
            "completion": f"{format_function_call_target(step)}{eos_token}",
        }


def iter_sft_records(steps: Iterable[TrajectoryStep]) -> Iterable[dict[str, str]]:
    """Legacy JSON-context export kept for artifact compatibility."""

    for step in steps:
        yield {
            "sample_id": step.sample_id,
            "prompt": json.dumps(step.context, ensure_ascii=False),
            "completion": format_function_call_target(step),
        }
