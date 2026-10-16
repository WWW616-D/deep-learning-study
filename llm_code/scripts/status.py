#!/usr/bin/env python
"""
One-glance status of the whole project.

Answers "where am I and what is on disk" without loading anything large.

    python scripts/status.py
    python scripts/status.py --runs-dir runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import resolve  # noqa: E402

OK = "[ok]"
NO = "[--]"
WARN = "[!!]"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    runs_dir = resolve(args.runs_dir)

    next_steps: list[str] = []

    # ---------------------------------------------------------------- env
    section("1. Environment")
    try:
        import torch

        print(f"  torch        : {torch.__version__}")
        print(f"  cuda build   : {torch.version.cuda}")
        print(f"  cuda avail   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            arch = torch.cuda.get_arch_list()
            sm = f"sm_{cc[0]}{cc[1]}"
            print(f"  gpu          : {torch.cuda.get_device_name(0)}")
            print(f"  capability   : {cc}  ({sm})")
            print(f"  {OK if sm in arch else WARN} {sm} present in arch_list: {sm in arch}")
            free, total = torch.cuda.mem_get_info()
            print(f"  vram free    : {human(free)} / {human(total)}")
        else:
            print(f"  {WARN} no CUDA device -- training will be 50-100x slower")
    except Exception as e:
        print(f"  {NO} torch import failed: {e}")

    for pkg in ("tokenizers", "numpy"):
        try:
            mod = __import__(pkg)
            print(f"  {pkg:<12} : {getattr(mod, '__version__', '?')}")
        except Exception:
            print(f"  {NO} {pkg} not importable")

    # -------------------------------------------------------------- data
    section("2. Raw data")
    raw_dirs = sorted((ROOT / "data/raw").glob("*")) if (ROOT / "data/raw").exists() else []
    if not raw_dirs:
        print(f"  {NO} no raw data")
        next_steps.append("python scripts/download_tinystories.py")
    for d in raw_dirs:
        if d.is_dir():
            files = sorted(d.glob("*"))
            print(f"  {d.relative_to(ROOT)}   ({human(dir_size(d))})")
            for f in files:
                if f.is_file():
                    print(f"      {f.name:<34} {human(f.stat().st_size)}")

    # --------------------------------------------------------- tokenizer
    section("3. Tokenizers")
    tok_dir = ROOT / "data/tokenizer"
    toks = sorted(tok_dir.glob("*.json")) if tok_dir.exists() else []
    toks = [t for t in toks if not t.name.endswith(".meta.json")]
    if not toks:
        print(f"  {NO} none trained")
        next_steps.append(
            "python scripts/prepare_pretrain_data.py tokenizer --preset tiny --out-prefix tiny"
        )
    for t in toks:
        meta_p = t.with_suffix(".meta.json")
        extra = ""
        if meta_p.exists():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                extra = f"  preset={m.get('preset', '?')}"
            except Exception:
                pass
        print(f"  {t.name:<30} {human(t.stat().st_size):>10}{extra}")
        try:
            from src.tokenizer import BPETokenizer

            tk = BPETokenizer.load(t)
            en = "The quick brown fox jumps over the lazy dog. " * 10
            zh = "这是一个用于测试分词压缩率的中文句子。" * 10
            print(
                f"      vocab={tk.vocab_size:,}  "
                f"tok/char en={len(tk.encode(en))/len(en):.3f} "
                f"zh={len(tk.encode(zh))/len(zh):.3f}"
            )
            if len(tk.encode(zh)) / len(zh) > 2.0:
                print(
                    f"      {WARN} Chinese costs ~{len(tk.encode(zh))/len(zh):.1f} tokens/char: "
                    "this vocabulary has no Chinese merges (English-only training data)"
                )
        except Exception as e:
            print(f"      {WARN} could not inspect: {e}")

    # ------------------------------------------------------------- bins
    section("4. Training binaries")
    proc = ROOT / "data/processed"
    bins = sorted(proc.glob("*.bin")) if proc.exists() else []
    if not bins:
        print(f"  {NO} none")
        next_steps.append(
            "python scripts/prepare_pretrain_data.py encode --preset tiny --out-prefix tiny"
        )
    for b in bins:
        meta_p = b.with_suffix(".meta.json")
        info = ""
        if meta_p.exists():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                info = f"  tokens={m.get('train_tokens', '?'):,}" if "train_tokens" in m else ""
            except Exception:
                pass
        print(f"  {b.name:<34} {human(b.stat().st_size):>10}{info}")

    # --------------------------------------------------------- sft data
    section("5. SFT data")
    sft = ROOT / "data/sft"
    shards = sorted(sft.glob("*.jsonl")) if sft.exists() else []
    if not shards:
        print(f"  {NO} none")
    else:
        total = 0
        for s in shards:
            n = sum(1 for line in s.open(encoding="utf-8") if line.strip())
            total += n
        print(f"  {len(shards)} files, {total:,} dialogues, {human(dir_size(sft))}")
        if total:
            next_steps.append("python scripts/validate_sft.py data/sft/*.jsonl")

    # ------------------------------------------------------------ runs
    section("6. Training runs")
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        print(f"  {NO} no runs yet")
        next_steps.append(
            "python -m src.train_pretrain --preset tiny --max-iters 600 "
            "--out-dir runs/tiny_smoke --micro-batch 32 --grad-accum 2"
        )
    else:
        from src.checkpoint import peek

        for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            print(f"\n  {run.name}   ({human(dir_size(run))})")
            ckpts = sorted(run.glob("*.pt"))
            if not ckpts:
                print(f"      {NO} no checkpoints")
            for c in ckpts:
                try:
                    m = peek(c)
                    print(
                        f"      {c.name:<22} {human(c.stat().st_size):>9}  "
                        f"kind={m.get('kind', '?'):<6} step={m.get('step', '?'):<7} "
                        f"best_val={m.get('best_val', float('nan')):.4f}"
                        if m.get("best_val") is not None
                        else f"      {c.name:<22} {human(c.stat().st_size):>9}  "
                        f"kind={m.get('kind', '?'):<6} step={m.get('step', '?')}"
                    )
                except Exception as e:
                    print(f"      {c.name:<22} {WARN} unreadable: {e}")
            log = run / "train_log.csv"
            if log.exists():
                lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
                print(f"      train_log.csv: {len(lines)-1} rows")
                print(f"        header : {lines[0]}")
                for ln in lines[-3:]:
                    print(f"        last   : {ln}")
                best = run / "best.pt"
                if best.exists():
                    next_steps.append(
                        f"python -m src.generate --ckpt {best} "
                        f"--tokenizer data/tokenizer/tiny_tokenizer.json "
                        f'--prompt "Once upon a time"'
                    )

    # ----------------------------------------------------------- disk
    section("7. Disk usage")
    total = 0
    for name in (".venv", "data/raw", "data/processed", "data/tokenizer", "data/sft",
                 "runs", "_tmp", "_sandbox"):
        p = ROOT / name
        if p.exists():
            s = dir_size(p)
            total += s
            print(f"  {name:<20} {human(s):>10}")
    print(f"  {'-'*20} {'-'*10}")
    print(f"  {'total (excl. parent repo)':<20} {human(total):>10}")

    # ------------------------------------------------------ next steps
    section("Suggested next commands")
    if next_steps:
        for i, s in enumerate(dict.fromkeys(next_steps), 1):  # dedupe, keep order
            print(f"  {i}. {s}")
    else:
        print("  everything looks built. try:")
        print('      python -m src.generate --prompt "Once upon a time" --interactive')
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
