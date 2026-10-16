#!/usr/bin/env python
"""
Validate synthesised SFT data before it ever reaches the trainer.

Never trust a generator's self-reported counts -- a teacher model that says
"25 lines written" may have written 22 parseable ones and three that silently
break your tokenizer. This script re-derives everything from the files.

Checks performed
----------------
  * every line parses as JSON
  * required keys, correct types
  * roles strictly alternate user/assistant, starting with user, ending with assistant
  * word-count limits per role
  * no control characters or stray newlines inside content
  * duplicate questions within a file AND across files
  * the chat template renders each sample without error, and round-trips

Usage
-----
    python scripts/validate_sft.py data/sft/part_*.jsonl
    python scripts/validate_sft.py --dir data/sft
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import template as tmpl  # noqa: E402
from src.paths import resolve  # noqa: E402


def norm(q: str) -> str:
    return " ".join(q.strip().lower().rstrip("?.!").split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="jsonl files (globs allowed)")
    ap.add_argument("--dir", help="directory to scan for *.jsonl")
    ap.add_argument("--max-user-words", type=int, default=25)
    ap.add_argument("--max-asst-words", type=int, default=60)
    ap.add_argument("--show-errors", type=int, default=8)
    args = ap.parse_args()

    paths: list[Path] = []
    for f in args.files:
        # resolve() anchors the glob to the project root, so the working
        # directory does not matter.
        paths.extend(Path(p) for p in sorted(glob.glob(str(resolve(f)))))
    if args.dir:
        paths.extend(sorted(resolve(args.dir).glob("*.jsonl")))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("no input files")
        return 2

    total = 0
    errors: list[tuple[str, int, str]] = []
    questions: list[str] = []
    per_file_questions: dict[str, list[str]] = defaultdict(list)
    user_words: list[int] = []
    asst_words: list[int] = []
    asst_chars: list[int] = []
    turn_pairs = Counter()
    systems = Counter()
    topics = Counter()
    valid_rows: list[dict] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                total += 1
                where = (path.name, lineno)

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append((*where, f"invalid JSON: {e.msg}"))
                    continue

                if not isinstance(obj, dict):
                    errors.append((*where, "top level is not an object"))
                    continue

                msgs = obj.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    errors.append((*where, "missing/empty 'messages'"))
                    continue

                ok = True
                for i, m in enumerate(msgs):
                    if not isinstance(m, dict):
                        errors.append((*where, f"message {i} is not an object"))
                        ok = False
                        break
                    role, content = m.get("role"), m.get("content")
                    if role not in ("user", "assistant"):
                        errors.append((*where, f"message {i} bad role {role!r}"))
                        ok = False
                        break
                    if not isinstance(content, str) or not content.strip():
                        errors.append((*where, f"message {i} empty/非字符串 content"))
                        ok = False
                        break
                    expect = "user" if i % 2 == 0 else "assistant"
                    if role != expect:
                        errors.append((*where, f"message {i} role {role!r}, expected {expect!r}"))
                        ok = False
                        break
                    if "\n" in content or "\r" in content or "\t" in content:
                        errors.append((*where, f"message {i} contains control whitespace"))
                        ok = False
                        break
                    words = len(content.split())
                    limit = args.max_user_words if role == "user" else args.max_asst_words
                    if words > limit:
                        errors.append((*where, f"message {i} ({role}) {words} words > {limit}"))
                        ok = False
                        break
                if not ok:
                    continue

                if msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
                    errors.append((*where, "does not start with user / end with assistant"))
                    continue

                # ---- template integration ----------------------------------
                system = obj.get("system") or tmpl.DEFAULT_SYSTEM
                try:
                    text = tmpl.build_prompt(msgs, system=system)
                    parsed = tmpl.parse_conversation(text)
                except Exception as e:
                    errors.append((*where, f"template failed: {e}"))
                    continue
                if len(parsed) != len(msgs) + 1:
                    errors.append((*where, f"template round-trip lost turns ({len(parsed)} vs {len(msgs)+1})"))
                    continue

                q = next(m["content"] for m in msgs if m["role"] == "user")
                questions.append(norm(q))
                per_file_questions[path.name].append(norm(q))
                for m in msgs:
                    (user_words if m["role"] == "user" else asst_words).append(len(m["content"].split()))
                    if m["role"] == "assistant":
                        asst_chars.append(len(m["content"]))
                turn_pairs[len(msgs) // 2] += 1
                systems[system] += 1
                topics[obj.get("topic", "")] += 1
                valid_rows.append(obj)

    # ---- report ---------------------------------------------------------
    print(f"files scanned : {len(paths)}")
    for p in paths:
        print(f"    {p.name:<20} {sum(1 for _ in p.open(encoding='utf-8')):>5} lines")
    print()
    print(f"total lines   : {total}")
    print(f"valid samples : {len(valid_rows)}")
    print(f"errors        : {len(errors)}")
    if errors and args.show_errors:
        print("  first few errors:")
        for name, lineno, msg in errors[: args.show_errors]:
            print(f"    {name}:{lineno}  {msg}")

    if not valid_rows:
        return 1

    uniq = len(set(questions))
    print()
    print(f"distinct questions   : {uniq} / {len(questions)}  "
          f"({len(questions)/max(uniq,1):.2f}x duplication)")
    worst = Counter(questions).most_common(5)
    if worst and worst[0][1] > 1:
        print("  most repeated:")
        for q, n in worst:
            if n > 1:
                print(f"    x{n:<3} {q[:66]}")

    # cross-file overlap is the thing a naive per-file check would miss
    cross = 0
    seen: dict[str, str] = {}
    for name, qs in per_file_questions.items():
        for q in qs:
            if q in seen and seen[q] != name:
                cross += 1
            seen.setdefault(q, name)
    print(f"cross-file duplicates: {cross}")

    print()
    print(f"user words      : min {min(user_words)}  med {st.median(user_words)}  max {max(user_words)}")
    print(f"assistant words : min {min(asst_words)}  med {st.median(asst_words)}  max {max(asst_words)}")
    print(f"assistant chars : med {st.median(asst_chars)}  max {max(asst_chars)}")
    print(f"turn pairs      : {dict(sorted(turn_pairs.items()))}")
    print(f"distinct systems: {len(systems)}")
    print("  system prompt distribution:")
    for s, n in systems.most_common():
        print(f"    {n:>5}  {s[:64]}")

    empty_topics = topics.get("", 0)
    if empty_topics:
        print(f"\nnote: {empty_topics} samples have no 'topic' field")

    # ---- verdict --------------------------------------------------------
    print()
    verdict_ok = not errors and len(valid_rows) == total
    print("VERDICT:", "PASS" if verdict_ok else "FAIL")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
