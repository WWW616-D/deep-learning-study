#!/usr/bin/env python
"""
Turn raw text into a tokenised .bin ready for pretraining.

Two stages, deliberately separate so you can inspect and re-run each one:

  tokenizer  train the byte-level BPE on a SAMPLE of the corpus
  encode     tokenise the whole corpus into a flat uint16 binary, then split
             off a validation slice

Sampling for tokenizer training is not a shortcut. BPE merge statistics
converge long before you have read 2 GB of text, and training on ~64 MB instead
of ~1900 MB turns a 20-minute job into seconds with no measurable difference.
The reported tokens/char is the number worth checking: if it is far from what
you expect, the vocabulary is wrong for the data's language mix.

Usage
-----
    # smoke-sized artefacts (a few minutes end to end)
    python scripts/prepare_pretrain_data.py all --preset tiny \
        --max-docs 200000 --out-prefix tiny

    # real run
    python scripts/prepare_pretrain_data.py all --preset base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PRESETS, get_preset  # noqa: E402
from src.dataset import bin_stats, dtype_for_vocab, iter_documents, prepare_bin, split_bin  # noqa: E402
from src.paths import resolve  # noqa: E402
from src.tokenizer import BPETokenizer, save_meta, train_from_files  # noqa: E402

DEFAULT_INPUT = ["data/raw/tinystories/TinyStories-train.txt"]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------


def cmd_tokenizer(args) -> int:
    cfg = get_preset(args.preset)
    paths = [Path(p) for p in args.input]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"ERROR: input not found: {missing}")
        print("  download first: python scripts/download_tinystories.py")
        return 2

    print(f"  preset      : {args.preset}  (vocab_size={cfg.vocab_size:,})")
    print(f"  input       : {[str(p) for p in paths]}")
    print(f"  sample      : {args.sample_mb} MB")
    print(f"  output      : {args.tokenizer}")
    print("-" * 68)

    t0 = time.time()
    tok = train_from_files(
        paths,
        vocab_size=cfg.vocab_size,
        show_progress=True,
        max_bytes=args.sample_mb * 1024 * 1024,
    )
    wrapper = BPETokenizer(tok)
    wrapper.save(args.tokenizer)
    save_meta(
        wrapper,
        Path(args.tokenizer).with_suffix(".meta.json"),
        preset=args.preset,
        trained_on_mb=args.sample_mb,
    )

    print(f"\n  trained in {time.time()-t0:.1f}s")
    print(f"  saved to {args.tokenizer}\n")
    print("  verification:")
    try:
        wrapper.verify()
    except AssertionError as e:
        print(f"  VERIFICATION FAILED:\n{e}")
        return 1
    return 0


def cmd_encode(args) -> int:
    tok = BPETokenizer.load(args.tokenizer)
    paths = [Path(p) for p in args.input]

    print(f"  tokenizer   : {args.tokenizer}  (vocab {tok.vocab_size:,})")
    print(f"  input       : {[str(p) for p in paths]}")
    print(f"  max_docs    : {args.max_docs or 'all'}")
    print(f"  output      : {args.out_bin}")
    print("-" * 68)

    t0 = time.time()
    report = prepare_bin(
        tok,
        paths,
        args.out_bin,
        max_docs=args.max_docs,
        batch_docs=args.batch_docs,
        separator=None if args.doc_separator in ("", "none", "newline") else args.doc_separator,
    )
    print(f"  encoded in {time.time()-t0:.1f}s")
    avg = report["tokens"] / max(report["documents"], 1)
    print(f"  average document length: {avg:.1f} tokens")
    if avg < 50:
        print(
            "  WARNING: documents average under 50 tokens. That usually means the\n"
            "           document separator is wrong for this corpus (see\n"
            "           --doc-separator); you may be tokenising fragments."
        )

    if report["tokens"] < args.val_tokens * 4:
        print(
            f"  skipping val split: only {report['tokens']:,} tokens, "
            f"need at least {args.val_tokens * 4:,}"
        )
        args.val_tokens = max(1024, report["tokens"] // 20)

    dtype = dtype_for_vocab(tok.vocab_size)
    train_n, val_n = split_bin(
        args.out_bin, args.out_train, args.out_val,
        val_tokens=args.val_tokens, dtype=dtype,
    )
    print(f"  split: train {train_n:,} tokens -> {args.out_train}")
    print(f"         val   {val_n:,} tokens -> {args.out_val}")

    # Write the token count next to the config, so the trainer and any later
    # analysis read one authoritative number instead of re-deriving it.
    meta_path = Path(args.out_train).with_suffix(".meta.json")
    meta = {
        "tokenizer": str(args.tokenizer),
        "vocab_size": tok.vocab_size,
        "dtype": str(dtype),
        "train_tokens": train_n,
        "val_tokens": val_n,
        "tokens_per_char": report["tokens_per_char"],
        "documents": report["documents"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  meta        : {meta_path}")

    print("\n  token content sanity:")
    for name, path in (("train", args.out_train), ("val", args.out_val)):
        st = bin_stats(path, dtype)
        print(
            f"    {name:5} : {st['tokens']:>12,} tokens  "
            f"distinct_in_sample={st.get('distinct_in_sample', '?')}  "
            f"most_common={st.get('most_common_fraction', 0):.4f}"
        )
    return 0


def cmd_report(args) -> int:
    tok = BPETokenizer.load(args.tokenizer)
    print(f"tokenizer: {args.tokenizer}")
    tok.verify()
    for name, path_str in (("train", args.out_train), ("val", args.out_val)):
        path = Path(path_str)
        if path.exists():
            st = bin_stats(path, dtype_for_vocab(tok.vocab_size))
            print(f"\n{name}: {path}  {human(path.stat().st_size)}")
            print(json.dumps(st, indent=2))
    return 0


# --------------------------------------------------------------------------


def build_common() -> argparse.ArgumentParser:
    """
    Options shared by every subcommand.

    Uses a parent parser rather than options on the top-level parser, so both
    `script.py --preset tiny encode` and `script.py encode --preset tiny` work.
    Putting them only on the top level silently rejects the second form, which
    is the form everyone types.
    """
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--preset", default="base", choices=sorted(PRESETS))
    c.add_argument("--input", nargs="*", default=list(DEFAULT_INPUT))
    c.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    c.add_argument("--out-bin", default=None, help="intermediate flat file")
    c.add_argument("--out-train", default=None)
    c.add_argument("--out-val", default=None)
    c.add_argument("--val-tokens", type=int, default=2_000_000)
    c.add_argument("--sample-mb", type=int, default=64,
                   help="how much of the corpus to read when training the tokenizer")
    c.add_argument("--max-docs", type=int, default=None,
                   help="cap the number of documents tokenised (for smoke tests)")
    c.add_argument("--batch-docs", type=int, default=5000)
    c.add_argument("--doc-separator", default="<|endoftext|>",
                   help="string that separates documents; use 'newline' for line-delimited corpora. "
                        "TinyStories uses <|endoftext|> and wraps each story over several lines.")
    c.add_argument("--out-prefix", default=None,
                   help="shorthand: <prefix>_tokenizer.json / <prefix>_train.bin / <prefix>_val.bin")
    return c


def main() -> int:
    common = build_common()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tokenizer", parents=[common], help="train the BPE vocabulary")
    sub.add_parser("encode", parents=[common], help="tokenise the corpus into .bin")
    sub.add_parser("report", parents=[common], help="inspect existing artefacts")
    sub.add_parser("all", parents=[common], help="tokenizer then encode")

    args = ap.parse_args()

    if args.out_prefix:
        p = args.out_prefix
        args.tokenizer = f"data/tokenizer/{p}_tokenizer.json"
        args.out_train = args.out_train or f"data/processed/{p}_train.bin"
        args.out_val = args.out_val or f"data/processed/{p}_val.bin"
        args.out_bin = args.out_bin or f"data/processed/{p}_all.bin"

    args.out_bin = args.out_bin or "data/processed/all.bin"
    args.out_train = args.out_train or "data/processed/train.bin"
    args.out_val = args.out_val or "data/processed/val.bin"

    # Anchor every path to the project root so the working directory (run.cmd,
    # PyCharm, plain python) never changes behaviour.
    args.tokenizer = str(resolve(args.tokenizer))
    args.out_bin = str(resolve(args.out_bin))
    args.out_train = str(resolve(args.out_train))
    args.out_val = str(resolve(args.out_val))
    args.input = [str(resolve(p)) for p in args.input]

    if args.cmd == "tokenizer":
        return cmd_tokenizer(args)
    if args.cmd == "encode":
        return cmd_encode(args)
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "all":
        rc = cmd_tokenizer(args)
        if rc:
            return rc
        print()
        return cmd_encode(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
