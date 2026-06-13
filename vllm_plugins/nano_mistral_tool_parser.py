"""Compatibility parser for Mistral v0.3 tool calls on NVIDIA vLLM 0.15.1."""

import json

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.mistral_tool_parser import MistralToolParser


@ToolParserManager.register_module("nano_mistral")
class NanoMistralToolParser(MistralToolParser):
    """Normalize recoverable pre-v11 tool calls before vLLM validates them."""

    def extract_tool_calls(self, model_output, request):
        if self._is_pre_v11 and self.bot_token in model_output:
            content_and_raw_tool_calls = model_output.split(self.bot_token)
            if len(content_and_raw_tool_calls) == 2:
                raw_tool_calls = content_and_raw_tool_calls[1].strip()
                try:
                    json.loads(raw_tool_calls)
                except json.JSONDecodeError:
                    matches = self.tool_call_regex.findall(raw_tool_calls)
                    if matches:
                        model_output = (
                            f"{content_and_raw_tool_calls[0]}{self.bot_token}{matches[0]}"
                        )

        return super().extract_tool_calls(model_output, request)
