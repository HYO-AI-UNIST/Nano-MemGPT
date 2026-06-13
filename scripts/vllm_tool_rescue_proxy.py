from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urljoin

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible proxy that rescues JSON-as-content tool calls."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--upstream", default=os.getenv("VLLM_UPSTREAM", "http://llama-vllm:8000"))
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def _tool_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    return function.get("name") if isinstance(function, dict) else None


def _tool_properties(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for tool in tools:
        if _tool_name(tool) != name:
            continue
        function = tool.get("function") or {}
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        return properties if isinstance(properties, dict) else {}
    return {}


def _balanced_objects(text: str) -> list[str]:
    objects: list[str] = []
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
        if character in {'"', "'"}:
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


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


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


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        folded = value.casefold()
        if folded == "true":
            return True
        if folded == "false":
            return False
    return value


def _looks_like_tool_result(value: dict[str, Any]) -> bool:
    if "status" in value and "message" in value:
        return True
    message = value.get("message")
    return isinstance(message, dict) and "results" in message


def _candidate_call(value: dict[str, Any], allowed_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    if _looks_like_tool_result(value):
        return None

    name = value.get("name")
    arguments = value.get("arguments", value.get("parameters"))
    if isinstance(name, str) and isinstance(arguments, dict):
        return name, dict(arguments)

    if "conversation_search" in allowed_names and isinstance(value.get("query"), str):
        return "conversation_search", dict(value)

    message = value.get("message")
    if "send_message" in allowed_names and isinstance(message, str):
        return "send_message", {"message": message}

    return None


def _sanitize_arguments(
    tools: list[dict[str, Any]],
    name: str,
    arguments: dict[str, Any],
    top_level: dict[str, Any],
) -> dict[str, Any] | None:
    properties = _tool_properties(tools, name)
    if not properties:
        return None

    cleaned = {key: value for key, value in arguments.items() if key in properties}
    if "request_heartbeat" in properties and "request_heartbeat" not in cleaned:
        heartbeat = top_level.get("request_heartbeat")
        cleaned["request_heartbeat"] = _coerce_bool(heartbeat) if heartbeat is not None else name != "send_message"
    if "thinking" in properties and "thinking" not in cleaned:
        thinking = top_level.get("thinking") or top_level.get("reason")
        if isinstance(thinking, str):
            cleaned["thinking"] = thinking
    if "roles" in cleaned:
        cleaned["roles"] = _coerce_roles(cleaned["roles"])
    if "limit" in cleaned:
        cleaned["limit"] = _coerce_int(cleaned["limit"])
    if "request_heartbeat" in cleaned:
        cleaned["request_heartbeat"] = _coerce_bool(cleaned["request_heartbeat"])

    if name == "conversation_search" and not isinstance(cleaned.get("query"), str):
        return None
    if name == "send_message" and not isinstance(cleaned.get("message"), str):
        return None
    return cleaned


def rescue_tool_call(content: str, request_body: dict[str, Any]) -> dict[str, Any] | None:
    tools = request_body.get("tools") or []
    if not isinstance(tools, list):
        return None
    allowed_names = {name for tool in tools if (name := _tool_name(tool))}
    if not allowed_names:
        return None

    candidates = []
    for text in _balanced_objects(content):
        value = _load_json_object(text)
        if value is None:
            continue
        call = _candidate_call(value, allowed_names)
        if call is None:
            continue
        name, arguments = call
        if name not in allowed_names:
            continue
        cleaned = _sanitize_arguments(tools, name, arguments, value)
        if cleaned is not None:
            candidates.append((name, cleaned))

    if len(candidates) != 1:
        return None

    name, arguments = candidates[0]
    return {
        "id": f"call_nano_rescue_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def rescue_response(response_body: dict[str, Any], request_body: dict[str, Any]) -> bool:
    rescued = False
    for choice in response_body.get("choices", []):
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        if message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        tool_call = rescue_tool_call(content, request_body)
        if tool_call is None:
            continue
        message["content"] = None
        message["tool_calls"] = [tool_call]
        message["function_call"] = None
        choice["finish_reason"] = "tool_calls"
        rescued = True
    return rescued


class RescueProxyHandler(BaseHTTPRequestHandler):
    server_version = "NanoMemGPTToolRescueProxy/0.1"

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None
        return json.loads(raw)

    def _send_json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        if headers:
            for key, value in headers.items():
                if key.casefold() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}:
                    self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _proxy_url(self) -> str:
        upstream = self.server.upstream.rstrip("/")  # type: ignore[attr-defined]
        return urljoin(upstream + "/", self.path.lstrip("/"))

    def do_GET(self) -> None:
        response = requests.get(self._proxy_url(), timeout=self.server.timeout)  # type: ignore[attr-defined]
        self._send_json(response.status_code, response.json(), dict(response.headers))

    def do_POST(self) -> None:
        try:
            request_body = self._read_json() or {}
            response = requests.post(
                self._proxy_url(),
                json=request_body,
                headers={
                    key: value
                    for key, value in self.headers.items()
                    if key.casefold() not in {"host", "content-length", "connection"}
                },
                timeout=self.server.timeout,  # type: ignore[attr-defined]
                stream=bool(request_body.get("stream")),
            )
            if request_body.get("stream"):
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.casefold() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}:
                        self.send_header(key, value)
                self.end_headers()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        self.wfile.write(chunk)
                return

            payload = response.json()
            rescued = False
            if self.path.rstrip("/") == "/v1/chat/completions" and isinstance(payload, dict):
                rescued = rescue_response(payload, request_body)
            if rescued:
                sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] rescued tool call\n")
                sys.stderr.flush()
            self._send_json(response.status_code, payload, dict(response.headers))
        except Exception as exc:
            self._send_json(502, {"error": {"message": f"{type(exc).__name__}: {exc}", "type": "proxy_error"}})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n")


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RescueProxyHandler)
    server.upstream = args.upstream
    server.timeout = args.timeout
    print(f"rescue proxy listening on {args.host}:{args.port}, upstream={args.upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
