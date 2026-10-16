"""
Smoke-test every command-line entry point.

Why this exists
---------------
The same argparse mistake was made twice in this project: shared options were
defined on the TOP-LEVEL parser instead of on a parent parser shared by the
subparsers. The result is that

    script.py <subcommand> --option value      <-- the form everyone types
      -> error: unrecognized arguments

    script.py --option value <subcommand>      <-- the form that works
      -> fine

Both orders are perfectly reasonable to type, so both must work. This test runs
the real scripts as subprocesses and fails if either order breaks.

Run:  python tests/test_cli.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (script, subcommands) -- subcommands may be empty for a plain script
ENTRY_POINTS = [
    ("src/train_pretrain.py", []),
    ("src/train_sft.py", []),
    ("src/generate.py", []),
    ("scripts/status.py", []),
    ("scripts/download_corpus.py", []),
    ("scripts/validate_sft.py", []),
    ("scripts/build_corpus.py", ["extract", "tokenizer", "encode", "mix", "all"]),
    ("scripts/prepare_pretrain_data.py", ["tokenizer", "encode", "report", "all"]),
    ("scripts/synth_sft_data.py", []),
    ("scripts/download_tinystories.py", []),
]

_failures: list[str] = []


def run(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def check(name: str, ok: bool, fail_detail: str = "") -> None:
    """Print PASS/FAIL. `fail_detail` is shown only when the check fails."""
    print(f"  {'[PASS]' if ok else '[FAIL]'} {name}" + (f"  -- {fail_detail}" if not ok and fail_detail else ""))
    if not ok:
        _failures.append(name)


def main() -> int:
    print("llm_code -- CLI entry point smoke tests\n")
    python = Path(sys.executable).name

    for script, subcommands in ENTRY_POINTS:
        path = ROOT / script
        if not path.exists():
            check(f"{script} exists", False, "file not found")
            continue

        print(f"\n{script}")

        rc, out = run([script, "--help"])
        check("  `--help` exits 0", rc == 0, f"rc={rc}" + (f"  {out.strip()[:80]}" if rc else ""))
        if rc != 0:
            continue

        for sub in subcommands:
            # order A: options after the subcommand (what people actually type)
            rc_a, out_a = run([script, sub, "--help"])
            check(f"  `{sub} --help` (options after subcommand) exits 0",
                  rc_a == 0, f"rc={rc_a}" + (f"  {out_a.strip()[:80]}" if rc_a else ""))

            # a bogus option must be REJECTED, not silently ignored
            rc_b, out_b = run([script, sub, "--definitely-not-an-option"])
            check(f"  `{sub}` rejects unknown options",
                  rc_b != 0, "unknown option was accepted")

            # and the same bogus option BEFORE the subcommand must also be rejected
            rc_c, out_c = run([script, "--definitely-not-an-option", sub])
            check(f"  `{sub}` rejects unknown options before the subcommand",
                  rc_c != 0, "unknown option was accepted")

    print("\n" + "=" * 68)
    if _failures:
        print(f"  {len(_failures)} check(s) FAILED:")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  All CLI entry points behave.")
    print(f"  (ran with {python}; pass the same interpreter to reproduce)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
