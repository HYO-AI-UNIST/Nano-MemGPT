"""Strict compatibility parsers for Nano-MemGPT tool-calling pilots.

These adapters only repackage explicit, schema-valid tool calls. They do not
choose a tool, repair arguments, or infer a call from natural-language text.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.llama_tool_parser import Llama3JsonToolParser
from vllm.tool_parsers.mistral_tool_parser import MistralToolParser


XML_FUNCTION_PATTERN = re.compile(
    r"""<function\s+
        name=(?P<quote>["'])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\1\s+
        arguments=(?P<args_quote>["'])(?P<arguments>\{.*?\})(?P=args_quote)
        (?:\s+request_heartbeat=(?P<hb_quote>["'])(?P<heartbeat>true|false)(?P=hb_quote))?
        \s*/>""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)
def _tool_name(tool: Any) -> str | None:
    function = getattr(tool, "function", None)
    if function is None and isinstance(tool, dict):
        function = tool.get("function")
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def _allowed_tool_names(request: Any) -> set[str]:
    return {
        name
        for tool in getattr(request, "tools", None) or []
        if (name := _tool_name(tool))
    }


def _tool_schema(request: Any, name: str) -> dict[str, Any] | None:
    for tool in getattr(request, "tools", None) or []:
        if _tool_name(tool) != name:
            continue
        function = getattr(tool, "function", None)
        if function is None and isinstance(tool, dict):
            function = tool.get("function")
        if isinstance(function, dict):
            return function.get("parameters")
        return getattr(function, "parameters", None)
    return None


def _schema_accepts(request: Any, name: str, arguments: dict[str, Any]) -> bool:
    schema = _tool_schema(request, name)
    return bool(schema and Draft202012Validator(schema).is_valid(arguments))


def _schema_properties(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _coerce_bool(value: Any) -> bool | Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        folded = value.casefold()
        if folded == "true":
            return True
        if folded == "false":
            return False
    return value


def _coerce_roles(value: Any) -> Any:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [role for role in value if isinstance(role, str)]
    return value


def _coerce_int(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _schema_rescue_arguments(
    request: Any, name: str, arguments: dict[str, Any]
) -> dict[str, Any] | None:
    """Prune schema-unknown keys and coerce simple transport-level types.

    This intentionally does not invent a query or answer. It only preserves
    fields that the model already emitted and removes wrapper noise that makes
    otherwise explicit tool-call JSON fail OpenAI tool schema validation.
    """

    schema = _tool_schema(request, name)
    properties = _schema_properties(schema)
    if not properties:
        return None

    rescued = {
        key: value
        for key, value in arguments.items()
        if key in properties
    }
    if "roles" in rescued:
        rescued["roles"] = _coerce_roles(rescued["roles"])
    if "limit" in rescued:
        rescued["limit"] = _coerce_int(rescued["limit"])
    if "request_heartbeat" in rescued:
        rescued["request_heartbeat"] = _coerce_bool(rescued["request_heartbeat"])
    if name != "send_message" and "request_heartbeat" in properties:
        rescued.setdefault("request_heartbeat", True)
    return rescued if _schema_accepts(request, name, rescued) else None


def _balanced_objects(text: str) -> list[str]:
    """Extract top-level brace-delimited objects while respecting quoted text."""

    objects = []
    start = None
    depth = 0
    quote = None
    escaped = False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _parse_object(text: str) -> dict[str, Any] | None:
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _json_like_calls(model_output: str) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    for text in _balanced_objects(model_output):
        value = _parse_object(text)
        if value is None:
            continue
        if "name" not in value:
            bare_arguments = value.get("arguments")
            if not isinstance(bare_arguments, dict):
                bare_arguments = value
            if isinstance(bare_arguments.get("message"), str):
                arguments = dict(bare_arguments)
                if "thinking" not in arguments and isinstance(value.get("thinking"), str):
                    arguments["thinking"] = value["thinking"]
                if "thinking" not in arguments:
                    arguments["thinking"] = "I have enough information to answer the user."
                calls.append(("send_message", arguments))
                continue
        name = value.get("name")
        arguments = value.get("arguments", value.get("parameters"))
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        if "arguments" in value and "parameters" in value:
            continue
        arguments = dict(arguments)
        if "thinking" not in arguments:
            if isinstance(value.get("thinking"), str):
                arguments["thinking"] = value["thinking"]
            elif isinstance(value.get("reason"), str):
                arguments["thinking"] = value["reason"]
            elif name == "send_message":
                arguments["thinking"] = "I have enough information to answer the user."
        heartbeat = value.get("request_heartbeat")
        if heartbeat is not None and "request_heartbeat" not in arguments:
            arguments["request_heartbeat"] = heartbeat
        elif name != "send_message" and "request_heartbeat" not in arguments:
            arguments["request_heartbeat"] = True
        calls.append((name, arguments))
    return calls


def _single_explicit_call(model_output: str, request: Any) -> ToolCall | None:
    """Return one explicit valid call, rejecting ambiguous or unknown calls."""

    calls = _json_like_calls(model_output)
    for match in XML_FUNCTION_PATTERN.finditer(model_output):
        try:
            arguments = json.loads(match.group("arguments"))
        except json.JSONDecodeError:
            continue
        heartbeat = match.groupdict().get("heartbeat")
        if heartbeat is not None and "request_heartbeat" not in arguments:
            arguments["request_heartbeat"] = heartbeat.casefold() == "true"
        calls.append((match.group("name"), arguments))

    if len(calls) != 1:
        return None
    name, arguments = calls[0]
    if name not in _allowed_tool_names(request) or not _schema_accepts(
        request, name, arguments
    ):
        return None

    return ToolCall(
        type="function",
        function=FunctionCall(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _single_rescued_call(model_output: str, request: Any) -> ToolCall | None:
    calls = _json_like_calls(model_output)
    if len(calls) != 1:
        return None
    name, arguments = calls[0]
    if name not in _allowed_tool_names(request):
        return None
    rescued = _schema_rescue_arguments(request, name, arguments)
    if rescued is None:
        return None
    return ToolCall(
        type="function",
        function=FunctionCall(
            name=name,
            arguments=json.dumps(rescued, ensure_ascii=False),
        ),
    )


class _StrictFallbackMixin:
    def extract_tool_calls(self, model_output, request):
        tool_call = _single_explicit_call(model_output, request)
        if tool_call is not None:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[tool_call],
                content=None,
            )

        parsed = super().extract_tool_calls(model_output, request)
        if parsed.tools_called:
            allowed_names = _allowed_tool_names(request)
            for tool_call in parsed.tool_calls:
                if tool_call.function.name not in allowed_names:
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=model_output,
                    )
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=model_output,
                    )
                if not isinstance(arguments, dict):
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=model_output,
                    )
                if not _schema_accepts(
                    request, tool_call.function.name, arguments
                ):
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=model_output,
                    )
            return parsed

        return parsed


class _RescueFallbackMixin(_StrictFallbackMixin):
    def extract_tool_calls(self, model_output, request):
        parsed = super().extract_tool_calls(model_output, request)
        if parsed.tools_called:
            return parsed

        tool_call = _single_rescued_call(model_output, request)
        if tool_call is not None:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[tool_call],
                content=None,
            )

        return parsed


@ToolParserManager.register_module("nano_strict_llama")
class NanoStrictLlamaToolParser(_StrictFallbackMixin, Llama3JsonToolParser):
    """Llama JSON parser with a conservative explicit-call fallback."""


@ToolParserManager.register_module("nano_strict_mistral")
class NanoStrictMistralToolParser(_StrictFallbackMixin, MistralToolParser):
    """Mistral parser with a conservative explicit-call fallback."""


@ToolParserManager.register_module("nano_rescue_llama")
class NanoRescueLlamaToolParser(_RescueFallbackMixin, Llama3JsonToolParser):
    """Llama parser that rescues explicit JSON calls with schema-noisy args."""


@ToolParserManager.register_module("nano_rescue_mistral")
class NanoRescueMistralToolParser(_RescueFallbackMixin, MistralToolParser):
    """Mistral parser that rescues explicit JSON calls with schema-noisy args."""
