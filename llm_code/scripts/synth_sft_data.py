#!/usr/bin/env python
"""
Synthesise SFT dialogues with a larger teacher model.

This is the step that turns a base model which "can talk" into one that "can
hold a conversation". A 31M-parameter student cannot absorb knowledge, so the
data has to be SHORT and NARROW: one domain, one or two turns, replies of a few
dozen words at most. Anything longer and the student learns to emit fluent
nonsense instead of learning to answer.

Design notes
------------
  * Batched generation. One API call produces BATCH dialogues, which amortises
    the (long) instruction prompt across all of them. This is the single
    biggest cost lever -- per-call prompting would cost roughly 5-10x more.
  * JSONL transport with lenient parsing. Teacher models love to wrap output in
    markdown fences or to emit a JSON array instead of lines. We accept all of
    it rather than throwing good data away.
  * Append-only output + resume. Free tiers rate-limit aggressively, so a run
    WILL be interrupted. Every accepted sample is flushed immediately and a
    rerun continues where it stopped instead of starting over.
  * Explicit validation. Cheap models routinely ignore length constraints, so
    over-long samples are dropped rather than hoped about.

Usage
-----
    python scripts/synth_sft_data.py --n 500 --batch 10

    # probe one call without writing anything
    python scripts/synth_sft_data.py --n 10 --batch 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.template import build_default_system_variants  # noqa: E402

# --------------------------------------------------------------------------
#  Domain
# --------------------------------------------------------------------------
# The student is 31M parameters. Narrowing the domain is not a limitation we
# tolerate, it is the technique that makes the model look competent at all.
#
# Topics are built as CONCEPT x ANGLE rather than listed flat. A flat list of
# 35 topics produced only 116 distinct questions across 500 samples (4.3x
# duplication) -- and, worse, 21 DIFFERENT answers to the same question, which
# teaches the student to be inconsistent. Crossing concepts with angles gives
# several hundred distinct angles cheaply.
CONCEPTS = [
    "adversarial examples",
    "the white-box threat model",
    "the black-box threat model",
    "the Fast Gradient Sign Method",
    "projected gradient descent attacks",
    "the perturbation budget epsilon",
    "L-infinity perturbation norms",
    "L2 perturbation norms",
    "transferability between models",
    "adversarial training",
    "the robustness-accuracy trade-off",
    "gradient masking",
    "JPEG compression as a defence",
    "input resizing as a defence",
    "feature squeezing",
    "randomized smoothing",
    "certified robustness",
    "adversarial patches",
    "targeted attacks",
    "untargeted attacks",
    "ensemble attacks",
    "decision-based black-box attacks",
    "query budgets in black-box attacks",
    "score-based black-box attacks",
    "data poisoning",
    "evasion attacks",
    "backdoor triggers",
    "membership inference",
    "model extraction",
    "universal adversarial perturbations",
    "adaptive attacks",
    "early stopping and reported robustness",
    "label smoothing with adversarial training",
    "the robustness Pareto frontier",
    "surrogate model selection in transfer studies",
    "one-step versus iterative attacks",
    "worst-case versus average-case accuracy",
    "physical-world adversarial examples",
    "detecting adversarial inputs",
    "data augmentation for robustness",
]

# All angles are noun phrases on purpose. Verb templates ("explain how {c} is
# evaluated") break on plural and mass-noun concepts -- "attacks is", "poisoning
# is not enough". Noun phrases are grammatical with any concept.
ANGLES = [
    "a one-sentence definition of {c}",
    "the practical importance of {c}",
    "the limits of {c}",
    "how to evaluate {c}",
    "a common misconception about {c}",
    "a comparison of {c} with a simpler alternative",
    "a practical limitation of {c}",
    "the practical cost of {c}",
]


def all_topics() -> list[str]:
    return [a.format(c=c) for c in CONCEPTS for a in ANGLES]

INSTRUCTION = """You are generating training data for a small language model.

Write {n} short English training dialogues about adversarial machine learning and AI security.

Cover these topics, roughly one dialogue each:
{topics}
{exclude}
Hard requirements -- these are not suggestions:
1. Output EXACTLY {n} lines. Each line is one JSON object and nothing else.
2. Each JSON object has exactly these keys:
     "system"   : the assistant's system prompt
     "messages" : a list of objects {{"role": ..., "content": ...}}, with roles
                  alternating "user" then "assistant", starting with "user" and
                  ending with "assistant".
3. All {n} questions in this batch must be DIFFERENT from each other. Do not
   ask the same thing twice in different words.
4. Exactly {multi} of the {n} dialogues must contain TWO user/assistant pairs.
   In those, the second user message is a natural follow-up to the first answer
   (asking for an example, a limitation, or a comparison). The remaining
   {one_turn} dialogues contain exactly ONE pair.
5. Each user message is ONE short question of at most 15 words.
6. Each assistant reply is at most 35 words. Plain prose only: no bullet lists,
   no markdown, no code, no equations, no LaTeX.
7. Replies must be factually correct and concrete. No filler, no hedging.
8. English only. Vary the phrasing; do not start every reply the same way.

One valid line, for reference:
{{"system": "You are a technical assistant. Explain things simply and accurately.", "messages": [{{"role": "user", "content": "What is an adversarial example?"}}, {{"role": "assistant", "content": "An input changed slightly so a model misclassifies it, while a human still sees the original class."}}]}}

Output only the {n} lines. No preamble, no explanation, no code fences."""


def build_exclude_block(questions: list[str], limit: int = 60) -> str:
    """
    List already-generated questions and forbid them.

    This is the mechanism that actually kills duplication. Telling a model to
    "be diverse" does nothing measurable; showing it the exact strings it has
    already produced does. Measured without this block: 500 samples contained
    only 116 distinct questions (4.3x duplication), with 21 DIFFERENT answers
    to the same question -- which trains the student to be inconsistent.
    """
    if not questions:
        return ""
    recent = questions[-limit:]
    body = "\n".join(f"- {q}" for q in recent)
    return (
        "\nThese questions are ALREADY in the dataset. Do NOT ask them again,\n"
        "and do not ask a reworded version of them:\n" + body + "\n"
    )


# --------------------------------------------------------------------------
#  Config
# --------------------------------------------------------------------------


def load_env(path: Path) -> None:
    """Minimal .env loader so the script needs no third-party packages."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:  # real env vars win over .env
            os.environ[k] = v


class CostTracker:
    def __init__(self, price_in: float | None, price_out: float | None) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.price_in = price_in
        self.price_out = price_out

    def add(self, usage: dict) -> None:
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    @property
    def cost(self) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (
            self.prompt_tokens / 1e6 * self.price_in
            + self.completion_tokens / 1e6 * self.price_out
        )

    def report(self) -> str:
        lines = [
            f"  API calls          : {self.calls}",
            f"  prompt tokens      : {self.prompt_tokens:,}",
            f"  completion tokens  : {self.completion_tokens:,}",
            f"  total tokens       : {self.prompt_tokens + self.completion_tokens:,}",
        ]
        c = self.cost
        if c is None:
            lines.append(
                "  estimated cost     : (set LLM_PRICE_IN / LLM_PRICE_OUT to compute)"
            )
        else:
            lines.append(
                f"  estimated cost     : {c:.4f} "
                f"(at {self.price_in}/M in, {self.price_out}/M out)"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
#  API
# --------------------------------------------------------------------------


def call_api(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None = None,
    retries: int = 6,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # A teacher model that "thinks" first burns most of its output budget on
    # invisible reasoning tokens. Measured on deepseek-flash, one request that
    # produced 116 useful output tokens cost 729 tokens with thinking enabled
    # (615 of them reasoning). For short, mechanical data generation none of
    # that reasoning is used, so it is pure waste -- 6x the bill for nothing.
    # reasoning_effort="none" disables it. Pass None to omit the field entirely
    # for endpoints that reject unknown parameters.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            # 429 and 5xx are expected on cheap/free tiers -- back off, do not die.
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = min(2 ** attempt * 2, 60)
                print(f"    HTTP {e.code}; retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise last_err
        except Exception as e:  # URLError, timeout, json decode
            last_err = e
            if attempt < retries - 1:
                wait = min(2 ** attempt * 2, 60)
                print(f"    {type(e).__name__}; retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last_err if last_err else RuntimeError("call_api failed")


# --------------------------------------------------------------------------
#  Parsing and validation
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_objects(text: str) -> list[dict]:
    """Pull JSON objects out of a teacher reply, tolerating common formatting."""
    text = text.strip()

    # 1. markdown fences
    fenced = _FENCE_RE.findall(text)
    if fenced:
        text = "\n".join(fenced)

    candidates: list[str] = []

    # 2. a single JSON array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [o for o in parsed if isinstance(o, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # 3. line-delimited JSON
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            candidates.append(line)

    # 4. last resort: brace matching over the whole blob
    if not candidates:
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])

    out: list[dict] = []
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def validate(obj: dict, max_user_words: int, max_asst_words: int) -> dict | None:
    """Return a cleaned sample, or None if it violates the contract."""
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None

    cleaned: list[dict] = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return None
        content = " ".join(content.split())
        if not content:
            return None
        cleaned.append({"role": role, "content": content})

    if cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        return None
    for i, m in enumerate(cleaned):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None

    for m in cleaned:
        words = len(m["content"].split())
        limit = max_user_words if m["role"] == "user" else max_asst_words
        if words > limit:
            return None

    system = obj.get("system")
    if not isinstance(system, str) or not system.strip():
        system = "You are a helpful assistant."
    system = " ".join(system.strip().split())[:200]

    return {"system": system, "messages": cleaned, "topic": obj.get("topic", "")}


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="target number of dialogues")
    ap.add_argument("--batch", type=int, default=10, help="dialogues per API call")
    ap.add_argument("--out", default="data/sft/raw.jsonl", help="output file")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topics-per-call", type=int, default=12)
    ap.add_argument("--multi-turn-ratio", type=float, default=0.3,
                    help="fraction of dialogues that should have two turns")
    ap.add_argument("--max-per-question", type=int, default=2,
                    help="hard cap on how often a single question may appear")
    ap.add_argument("--max-user-words", type=int, default=25)
    ap.add_argument("--max-asst-words", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = auto-size from --batch")
    ap.add_argument("--reasoning-effort", default="none",
                    help="'none' disables teacher reasoning tokens (big saving); "
                         "'omit' sends no such field")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--dry-run", action="store_true", help="one call, print samples, write nothing")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    load_env(ROOT / ".env")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("LLM_MODEL", "")
    if not (api_key and base_url and model):
        print("ERROR: set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (see .env.example)")
        return 2

    price_in = os.environ.get("LLM_PRICE_IN") or None
    price_out = os.environ.get("LLM_PRICE_OUT") or None
    tracker = CostTracker(
        float(price_in) if price_in else None, float(price_out) if price_out else None
    )

    # With reasoning disabled a dialogue costs roughly 110 output tokens, so
    # batch*250 leaves comfortable headroom without inviting runaway output.
    max_tokens = args.max_tokens or max(1000, args.batch * 250)
    reasoning_effort = (
        None
        if args.reasoning_effort.strip().lower() in ("omit", "off", "")
        else args.reasoning_effort.strip()
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- resume ---------------------------------------------------------
    existing: list[dict] = []
    if out_path.exists() and not args.dry_run:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if existing:
            print(f"resuming: {len(existing)} samples already in {args.out}")

    rng = random.Random(args.seed)
    systems = build_default_system_variants()
    topic_pool = all_topics()

    # Questions already emitted, oldest first. Used for two things: telling the
    # teacher what NOT to ask again, and enforcing a hard per-question cap. Both
    # are needed -- the prompt reduces duplication, the cap guarantees it.
    used_questions: list[str] = []
    question_counts: dict[str, int] = {}
    for r in existing:
        for m in r.get("messages", []):
            if m.get("role") == "user":
                q = str(m.get("content", "")).strip().lower()
                used_questions.append(q)
                question_counts[q] = question_counts.get(q, 0) + 1

    print(f"endpoint : {base_url}")
    print(f"model    : {model}")
    print(f"target   : {args.n} dialogues, {args.batch} per call")
    print(f"output   : {args.out}")
    print("-" * 68)

    accepted = 0
    rejected_long = 0
    rejected_bad = 0
    rejected_dup = 0
    multi_turn_made = 0
    call_no = 0

    # How many of each batch must be two-turn. Asking for "1 or 2 pairs" was
    # ignored completely (500/500 came back single-turn), so the number is now
    # explicit and counted.
    multi_turn_quota = (
        min(max(1, round(args.batch * args.multi_turn_ratio)), args.batch - 1)
        if args.batch > 1
        else 0
    )

    mode = "a" if (existing and not args.dry_run) else "w"
    fh = None if args.dry_run else out_path.open(mode, encoding="utf-8")

    try:
        while accepted < args.n:
            call_no += 1
            topics = rng.sample(topic_pool, min(args.topics_per_call, len(topic_pool)))
            prompt = INSTRUCTION.format(
                n=args.batch,
                topics="\n".join(f"- {t}" for t in topics),
                exclude=build_exclude_block(used_questions),
                multi=multi_turn_quota,
                one_turn=args.batch - multi_turn_quota,
            )

            t0 = time.time()
            try:
                resp = call_api(
                    base_url, api_key, model, prompt,
                    max_tokens, args.temperature, reasoning_effort,
                )
            except Exception as e:
                print(f"  call {call_no}: FAILED after retries -- {e}")
                break

            usage = resp.get("usage") or {}
            tracker.add(usage)
            text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            objs = extract_json_objects(text)

            batch_accepted = 0
            for i, obj in enumerate(objs):
                if accepted + batch_accepted >= args.n:
                    break
                sample = validate(obj, args.max_user_words, args.max_asst_words)
                if sample is None:
                    # Distinguish "too long" from "malformed" so the numbers tell
                    # you WHICH prompt constraint the teacher keeps ignoring.
                    msgs = obj.get("messages")
                    if isinstance(msgs, list) and msgs:
                        words = [
                            len(str(m.get("content", "")).split())
                            for m in msgs
                            if isinstance(m, dict)
                        ]
                        if words and max(words) > args.max_asst_words:
                            rejected_long += 1
                            continue
                    rejected_bad += 1
                    continue

                question = next(
                    m["content"] for m in sample["messages"] if m["role"] == "user"
                ).strip().lower()
                if question_counts.get(question, 0) >= args.max_per_question:
                    rejected_dup += 1
                    continue
                question_counts[question] = question_counts.get(question, 0) + 1
                used_questions.append(question)

                if not sample.get("topic"):
                    sample["topic"] = topics[i % len(topics)]
                sample["system"] = systems[rng.randrange(len(systems))]
                if len(sample["messages"]) >= 4:
                    multi_turn_made += 1

                batch_accepted += 1
                if args.dry_run:
                    print(json.dumps(sample, ensure_ascii=False))
                else:
                    assert fh is not None
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fh.flush()  # survive an interrupt

            accepted += batch_accepted
            dt = time.time() - t0
            cost = tracker.cost
            cost_str = f" cost~{cost:.4f}" if cost is not None else ""
            print(
                f"  call {call_no:>3}  +{batch_accepted:>2} ok / "
                f"{len(objs):>2} parsed  |  total {accepted}/{args.n}  "
                f"({dt:.1f}s){cost_str}"
            )

            if args.dry_run:
                break
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        if fh is not None:
            fh.close()

    # ---- summary --------------------------------------------------------
    print("=" * 68)
    print("token usage")
    print(tracker.report())
    print()
    print(f"  accepted samples    : {accepted}" + ("  (new this run)" if existing else ""))
    print(f"  rejected (too long) : {rejected_long}")
    print(f"  rejected (malformed): {rejected_bad}")
    print(f"  rejected (duplicate): {rejected_dup}")
    print(f"  multi-turn dialogues: {multi_turn_made}")
    print(f"  distinct questions  : {len(question_counts)}")
    total_rej = rejected_long + rejected_bad
    if accepted + total_rej:
        print(f"  accept rate        : {accepted / (accepted + total_rej) * 100:.1f}%")
    print(f"  output file        : {out_path}")
    if existing and not args.dry_run:
        total_on_disk = len(existing) + accepted
        print(f"  total on disk      : {total_on_disk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
