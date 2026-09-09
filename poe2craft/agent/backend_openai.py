"""OpenAI-compatible backend (DeepSeek, LM Studio, vLLM, ...).

Manual tool loop over `chat.completions` with streaming.  DeepSeek specifics:
thinking is on by default and returns `reasoning_content`; when tools are in
play that field MUST be echoed back in the assistant messages of the chain, or
the API answers 400.  We therefore keep it in history unconditionally.
"""
from __future__ import annotations

import json
import os
import sys

from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError

MAX_TOOL_RESULT_CHARS = 40000


def make_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL")
    if not base and os.environ.get("DEEPSEEK_API_KEY"):
        base = "https://api.deepseek.com"
    if not key:
        raise RuntimeError("set DEEPSEEK_API_KEY (or OPENAI_API_KEY + OPENAI_BASE_URL)")
    return OpenAI(api_key=key, base_url=base)


def openai_tools(tools) -> list[dict]:
    out = []
    for t in tools:
        d = t.to_dict()
        out.append({"type": "function", "function": {
            "name": d["name"], "description": d.get("description", ""),
            "parameters": d["input_schema"]}})
    return out


def _grey(s: str) -> None:
    print(f"\033[90m{s}\033[0m", file=sys.stderr, flush=True)


def run_turn(client: OpenAI, model: str, messages: list, tools, effort: str = "high",
             max_iterations: int = 24, max_tokens: int = 8192, verbose: bool = True) -> str:
    """messages: OpenAI-format history (system first). Appends this turn's messages in place."""
    by_name = {t.name: t for t in tools}
    oa_tools = openai_tools(tools)
    extra = {"reasoning_effort": effort} if effort else {}
    final_text = ""
    usage_in = usage_out = 0

    for _ in range(max_iterations):
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=oa_tools, stream=True,
            max_tokens=max_tokens, stream_options={"include_usage": True}, extra_body=extra)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict] = {}
        finish = None
        thinking_shown = False
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_in += chunk.usage.prompt_tokens or 0
                usage_out += chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            d = ch.delta
            rc = getattr(d, "reasoning_content", None)
            if rc:
                reasoning_parts.append(rc)
                if verbose and not thinking_shown:
                    print("\033[90m…thinking\033[0m", end="", file=sys.stderr, flush=True)
                    thinking_shown = True
            if d.content:
                if thinking_shown and not text_parts:
                    print("\r" + " " * 12 + "\r", end="", file=sys.stderr, flush=True)
                text_parts.append(d.content)
                if verbose:
                    print(d.content, end="", flush=True)
            for tc in d.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
            if ch.finish_reason:
                finish = ch.finish_reason

        text = "".join(text_parts)
        assistant: dict = {"role": "assistant", "content": text or None}
        rc_all = "".join(reasoning_parts)
        if rc_all:
            assistant["reasoning_content"] = rc_all
        tool_calls = [calls[i] for i in sorted(calls)]
        if tool_calls:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}}
                                       for c in tool_calls]
        messages.append(assistant)
        if text:
            final_text = text
            if verbose:
                print(flush=True)

        if not tool_calls:
            if finish == "length" and verbose:
                _grey("[cut off at max_tokens]")
            break

        for c in tool_calls:
            tool = by_name.get(c["name"])
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError as e:
                args, result = None, f"error: tool arguments are not valid JSON ({e})"
            else:
                result = None
            if verbose:
                a = c["arguments"] if len(c["arguments"]) <= 300 else c["arguments"][:300] + "…"
                _grey(f"⚙ {c['name']}({a})")
            if result is None:
                if tool is None:
                    result = f"error: unknown tool {c['name']!r}"
                else:
                    try:
                        result = tool.call(args)
                    except Exception as e:      # tools must never kill the loop
                        result = f"error: {type(e).__name__}: {e}"
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)
            if len(result) > MAX_TOOL_RESULT_CHARS:
                result = result[:MAX_TOOL_RESULT_CHARS] + "\n[... truncated; use a narrower filter]"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    if verbose:
        _grey(f"[tokens: {usage_in} in / {usage_out} out]")
    return final_text
