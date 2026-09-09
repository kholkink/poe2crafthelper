"""Interactive PoE2 assistant: an LLM + the poe2craft tools.

    python -m poe2craft.agent.chat                 # REPL
    python -m poe2craft.agent.chat "как скрафтить ботинки с MS и лайфом?"   # one question

Providers (auto-detected from the environment or ROOT/.env):
  ANTHROPIC_API_KEY                      -> Claude (claude-opus-5), tool runner, adaptive thinking
  DEEPSEEK_API_KEY                       -> DeepSeek (deepseek-v4-flash), OpenAI-compatible loop
  OPENAI_API_KEY + OPENAI_BASE_URL       -> any OpenAI-compatible server (LM Studio, vLLM, ...)
Override with --provider / --model / POE2_AGENT_MODEL / POE2_AGENT_EFFORT.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

from . import DEFAULT_LEAGUE, ROOT
from .tools import TOOLS

SYSTEM = """You are a Path of Exile 2 expert assistant (Early Access 0.5.x, late 2026) with tools:
a local knowledge base (poe2wiki + official patch notes), game data (bases, modifier pools, tiers,
spawn-weight estimates), live poe.ninja prices, the official trade search, and a crafting solver
that computes the cheapest expected-cost policy for a target item.

Language: answer in the language the user writes in (Russian -> Russian). Keep item, currency,
modifier and mechanic names in English (optionally with a Russian gloss the first time).

Ground rules
- Your memorised knowledge of PoE2 is stale. Every number (prices, tiers, weights, drop
  sources, patch changes) and every mechanic claim comes from a tool call. If tools return
  nothing, say so; do not fill gaps from memory.
- Cite the source URL for facts taken from search_knowledge / read_page.
- Be concise and concrete: tables and numbered steps, no filler, no restating the question.
- State assumptions you made (league, ilvl, budget, tier ranges) instead of asking many
  questions; ask only when the answer really changes the plan (e.g. which slot/base).

Crafting workflow
1. Map the wish to a base: find_bases (highest drop level is usually right). Default ilvl 82
   unless the mods need more; T1 life etc. may need ilvl 75-82, check list_mod_families.
2. list_mod_families with `contains` filters to get exact family ids and tiers. Translate the
   user's words ("MS", "лайф", "резы", "атака") into families. Mods that only come from
   desecration (dual chaos resists, +max res, ...) are in list_desecrated_families.
3. For each requirement, look for an essence that guarantees it: search_knowledge
   "essence <stat>" and pass its name plus essence_tier (compare the essence's value with the
   family's tier table). Lesser/normal/Greater essences work on magic items; Perfect essences
   on rares (perfect_essence field).
4. plan_craft. It models the whole 0.5 toolbox (Greater/Perfect orbs, Whittling, Erasure,
   Fracturing, desecration, essences) and picks what is worth its price. If the player already
   owns an item or found a listing (e.g. a fractured base), pass it as `start` with its price
   as base_price_ex. Restrict `mechanics` only when the user says what they can afford or asks
   to compare strategies ("without omens", "only basic currency").
5. Present: expected cost in divines and exalts as a RANGE (+/-50%), the shopping list
   (expected uses x unit price from expected_spend), and a step-by-step recipe written as
   decisions: "apply X; if you hit A -> do B; if you get a wrong tier of C -> chaos/reset".
   Use the decision_table (ordered by how often the state is visited). Name the exact
   currency variant the solver chose (e.g. Greater Orb of Transmutation, Preserved Rib +
   Omen of Dextral Necromancy) and say why it beats the plain one when that is not obvious.
6. Mention the caveats that matter for this craft (weights are estimates; whittling and
   desecration offer weights are approximations; Hinekora's Lock / recombination not modelled).
7. Offer trade_search for the base (or a fractured/partially rolled one) when a price
   matters; use at most one or two searches per answer.

Other questions (farming uniques, mechanics, builds)
- Uniques: read_page for the item; report drop source / boss / area from its acquisition
  section, then trade_search with name + rarity "unique" for the price.
- Build advice: you have no damage calculator. Reason qualitatively from the knowledge base
  and say explicitly that numbers are not computed; suggest what to check in Path of Building.
"""


def clean_text(q: str) -> str:
    """Repair terminal input that Python decoded with the wrong codec.

    A non-UTF-8 locale (LANG=C) or a cp1251 terminal leaves lone surrogates
    (\udcd0...) in the string; json/httpx then fail with 'surrogates not allowed'.
    Recover the original bytes and decode them as UTF-8, then cp1251, else drop them.
    """
    try:
        q.encode("utf-8")
        return q
    except UnicodeEncodeError:
        raw = q.encode("utf-8", "surrogateescape")
        for enc in ("utf-8", "cp1251"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                pass
        return q.encode("utf-8", "replace").decode("utf-8")


def setup_streams() -> None:
    """UTF-8 in, never crash on output: applies regardless of the shell locale."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="surrogateescape")
    except (AttributeError, ValueError):
        pass


def load_env(path: str = os.path.join(ROOT, ".env")) -> None:
    """Minimal KEY=VALUE loader; never overrides variables already set."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def detect_provider() -> str | None:
    if os.environ.get("POE2_AGENT_PROVIDER"):
        return os.environ["POE2_AGENT_PROVIDER"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


class Backend:
    """One conversation against one provider; `ask()` runs a full tool loop for a user turn."""

    def __init__(self, provider: str, model: str | None, effort: str):
        self.provider = provider
        self.effort = effort
        self.messages: list = []
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = model or "claude-opus-5"
        elif provider == "openai":
            from . import backend_openai
            self.client = backend_openai.make_client()
            self.model = model or os.environ.get("POE2_AGENT_MODEL") or (
                "deepseek-v4-flash" if os.environ.get("DEEPSEEK_API_KEY") else "default")
            self.messages.append({"role": "system", "content": SYSTEM})
        else:
            raise ValueError(f"unknown provider {provider!r}")

    def ask(self, q: str) -> str:
        self.messages.append({"role": "user", "content": q})
        if self.provider == "anthropic":
            return self._ask_anthropic()
        from . import backend_openai
        return backend_openai.run_turn(self.client, self.model, self.messages, TOOLS, effort=self.effort)

    def _ask_anthropic(self) -> str:
        runner = self.client.beta.messages.tool_runner(
            model=self.model, max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"}, output_config={"effort": self.effort},
            tools=TOOLS, messages=self.messages, max_iterations=24, stream=True,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        final_text = ""
        for stream in runner:
            parts = []
            with stream:
                for event in stream:
                    if event.type == "text":
                        parts.append(event.text)
                        print(event.text, end="", flush=True)
                msg = stream.get_final_message()
            self.messages.append({"role": "assistant", "content": msg.content})
            if msg.stop_reason == "refusal":
                print("\n[model declined this request]", file=sys.stderr)
                break
            for b in msg.content:
                if b.type == "tool_use":
                    a = json.dumps(b.input, ensure_ascii=False)
                    print(f"\n\033[90m⚙ {b.name}({a[:300]})\033[0m", file=sys.stderr, flush=True)
            resp = runner.generate_tool_call_response()
            if resp is not None:
                self.messages.append(resp)
            final_text = "".join(parts) or final_text
        print()
        return final_text

    def reset(self) -> None:
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    def dump(self) -> list:
        out = []
        for m in self.messages:
            c = m.get("content")
            if isinstance(c, list):
                c = [(b.to_dict() if hasattr(b, "to_dict") else b) for b in c]
            out.append({**m, "content": c})
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="one-shot question (omit for a REPL)")
    ap.add_argument("--provider", choices=["anthropic", "openai"])
    ap.add_argument("--model")
    ap.add_argument("--effort", default=os.environ.get("POE2_AGENT_EFFORT", "high"),
                    help="low/medium/high/xhigh/max for Claude; low/high/max for DeepSeek")
    ap.add_argument("--save", help="append each turn's full conversation to this JSONL file")
    a = ap.parse_args(argv)

    setup_streams()
    load_env()
    provider = a.provider or detect_provider()
    if provider is None:
        sys.exit("no API key found: set ANTHROPIC_API_KEY, DEEPSEEK_API_KEY or OPENAI_API_KEY "
                 f"(env or {os.path.join(ROOT, '.env')})")
    try:
        be = Backend(provider, a.model or os.environ.get("POE2_AGENT_MODEL"), a.effort)
    except Exception as e:
        sys.exit(f"cannot create client: {e}")
    print(f"poe2craft agent · {provider}/{be.model} · league {DEFAULT_LEAGUE} · effort {a.effort}", file=sys.stderr)

    def ask(q: str) -> None:
        q = clean_text(q)
        try:
            be.ask(q)
        except KeyboardInterrupt:
            print("\n[interrupted]", file=sys.stderr)
        except Exception as e:
            print(f"\n[{type(e).__name__}] {e}", file=sys.stderr)
        if a.save:
            with open(a.save, "a", encoding="utf-8") as f:
                f.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"),
                                    "provider": provider, "model": be.model, "q": q,
                                    "messages": be.dump()}, ensure_ascii=False, default=str) + "\n")

    if a.question:
        ask(clean_text(" ".join(a.question)))
        return
    print("Ask about PoE2 crafting, mechanics, farming, prices. Ctrl-D or /quit to exit, /reset to forget history.",
          file=sys.stderr)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            be.reset()
            print("(history cleared)", file=sys.stderr)
            continue
        ask(q)


if __name__ == "__main__":
    main()
