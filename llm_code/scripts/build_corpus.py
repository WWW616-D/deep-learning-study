#!/usr/bin/env python
"""
Build the pretraining corpus from downloaded parquet files.

Five stages, each runnable alone so you can inspect between them:

    extract    parquet -> plain text, one file per source
    tokenizer  train one BPE vocabulary on a sample of ALL sources at once
    encode     tokenise each source into its own .bin
    mix        interleave the .bins at a target ratio -> train.bin / val.bin
    all        all of the above

Why the tokenizer is trained on a MIXED sample
----------------------------------------------
Vocabulary merges are shared, and the vocabulary size is fixed. If you train it
on English only, Chinese costs ~3 tokens per character (each character falls
back to raw UTF-8 bytes) and the model's effective context collapses. Training
on both makes the merge budget split sensibly.

Why sources are encoded separately and then interleaved
-------------------------------------------------------
Mixing by document count is wrong, because an English web page and a Chinese
article differ wildly in length. Encoding each source separately and then
interleaving fixed-size token chunks gives exact control over the ratio, and the
result is trivially verifiable: count the tokens from each side in the output.

Usage
-----
    python scripts/build_corpus.py extract
    python scripts/build_corpus.py tokenizer --preset base --sample-mb 200
    python scripts/build_corpus.py encode --preset base
    python scripts/build_corpus.py mix --target-tokens 630000000 --ratio 2:1
    python scripts/build_corpus.py all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import PRESETS, get_preset  # noqa: E402
from src.dataset import dtype_for_vocab  # noqa: E402
from src.net import human  # noqa: E402
from src.paths import resolve  # noqa: E402
from src.tokenizer import BPETokenizer, save_meta, train_tokenizer  # noqa: E402

# Source registry: name -> (parquet dir, extracted text file)
SOURCES = {
    "en": ("data/raw/fineweb-edu", "data/raw/corpus_en.txt"),
    "zh": ("data/raw/wikipedia-zh", "data/raw/corpus_zh.txt"),
}


# --------------------------------------------------------------------------
#  extract
# --------------------------------------------------------------------------


def iter_parquet_text(path: Path, column: str = "text", batch_rows: int = 8192):
    """
    Stream the text column out of a parquet file.

    Parquet is read in row-group batches rather than loaded whole: a 2 GB file
    holds hundreds of megabytes of Python strings once decoded, and there is no
    reason to hold all of it at once.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    if column not in pf.schema_arrow.names:
        raise ValueError(
            f"{path.name} has no column {column!r}; available: {pf.schema_arrow.names}"
        )
    for batch in pf.iter_batches(batch_size=batch_rows, columns=[column]):
        for v in batch.column(column).to_pylist():
            if v:
                yield v


def cmd_extract(args) -> int:
    for key, (src_dir, out_file) in SOURCES.items():
        if args.only and key not in args.only:
            continue
        src = resolve(src_dir)
        out = resolve(out_file)
        if not src.exists():
            print(f"[{key}] SKIP: {src} does not exist")
            continue
        if out.exists() and not args.force:
            print(f"[{key}] {out.name} already exists ({human(out.stat().st_size)}), "
                  f"use --force to redo")
            continue

        files = sorted(src.glob("*.parquet"))
        if not files:
            print(f"[{key}] SKIP: no parquet files in {src}")
            continue

        print(f"[{key}] {len(files)} parquet file(s) -> {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n_docs = 0
        n_chars = 0
        with out.open("w", encoding="utf-8") as fh:
            for f in files:
                for text in iter_parquet_text(f):
                    # One document per line; newlines inside a document would
                    # break the line-based reader everywhere downstream.
                    text = " ".join(text.split())
                    if len(text) < 50:
                        continue
                    fh.write(text + "\n")
                    n_docs += 1
                    n_chars += len(text)
                print(f"      {f.name}: cumulative {n_docs:,} docs, {human(n_chars)}",
                      flush=True)
        print(f"[{key}] done: {n_docs:,} docs, {n_chars:,} chars, "
              f"{human(out.stat().st_size)} in {time.time()-t0:.0f}s\n")
    return 0


# --------------------------------------------------------------------------
#  tokenizer
# --------------------------------------------------------------------------


def cmd_tokenizer(args) -> int:
    cfg = get_preset(args.preset)
    files = [resolve(p) for _, p in SOURCES.values()]
    files = [f for f in files if f.exists()]
    if not files:
        print("ERROR: no extracted text files. Run `extract` first.")
        return 2

    print(f"  vocab size : {cfg.vocab_size:,}")
    print(f"  sources    : {[f.name for f in files]}")
    print(f"  sample     : {args.sample_mb} MB per source")
    print("-" * 70)

    def sample_iter():
        """
        Yield decoded text, counting SAMPLE BYTES correctly.

        Opening in text mode and calling fh.read(n) counts CHARACTERS, not bytes.
        For Chinese that is a 2.5x undercount (one character is 2-3 UTF-8 bytes),
        so a "200 MB" sample silently became ~500 MB -- and BPE training on
        ~700 MB of mixed text does not finish in a reasonable time. Reading
        binary and decoding after counting gets the size right.
        """
        for f in files:
            with f.open("rb") as fh:
                left = args.sample_mb * 1024 * 1024
                while left > 0:
                    raw = fh.read(min(1 << 20, left))
                    if not raw:
                        break
                    left -= len(raw)
                    yield raw.decode("utf-8", errors="replace")

    total_mb = args.sample_mb * len(files)
    print(f"  feeding ~{total_mb} MB of text to the BPE trainer "
          f"(this is I/O + merge counting; expect tens of seconds)")

    t0 = time.time()
    tok = train_tokenizer(sample_iter(), vocab_size=cfg.vocab_size, show_progress=True)
    wrapper = BPETokenizer(tok)
    out = resolve(args.tokenizer)
    wrapper.save(out)
    save_meta(wrapper, out.with_suffix(".meta.json"), preset=args.preset,
              trained_on="mixed sample")

    print(f"\n  trained in {time.time()-t0:.1f}s -> {out}\n  verification:")
    try:
        rep = wrapper.verify()
    except AssertionError as e:
        print(f"  VERIFICATION FAILED:\n{e}")
        return 1
    if rep["tokens_per_char_zh"] > 1.2:
        print(
            f"  [!!] Chinese still costs {rep['tokens_per_char_zh']:.2f} tokens/char. "
            "That usually means the Chinese sample was too small relative to English."
        )
    return 0


# --------------------------------------------------------------------------
#  encode
# --------------------------------------------------------------------------


def cmd_encode(args) -> int:
    tok = BPETokenizer.load(resolve(args.tokenizer))
    dtype = dtype_for_vocab(tok.vocab_size)

    from src.dataset import prepare_bin

    for key, (_, txt) in SOURCES.items():
        p = resolve(txt)
        if not p.exists():
            print(f"[{key}] SKIP: {p} missing")
            continue
        out = resolve(f"data/processed/src_{key}.bin")
        if out.exists() and not args.force:
            print(f"[{key}] {out.name} exists ({human(out.stat().st_size)}), skipping")
            continue
        print(f"[{key}] encoding {p.name} ({human(p.stat().st_size)})")
        t0 = time.time()
        # separator=None: `extract` already wrote one document per line.
        rep = prepare_bin(tok, [p], out, separator=None, batch_docs=args.batch_docs)
        avg = rep["tokens"] / max(rep["documents"], 1)
        print(f"      {time.time()-t0:.0f}s   avg {avg:.0f} tokens/doc   "
              f"{rep['tokens_per_char']:.3f} tokens/char\n")
    return 0


# --------------------------------------------------------------------------
#  mix
# --------------------------------------------------------------------------


def cmd_mix(args) -> int:
    tok = BPETokenizer.load(resolve(args.tokenizer))
    dtype = dtype_for_vocab(tok.vocab_size)

    parts = args.ratio.split(":")
    if len(parts) != 2:
        print("ERROR: --ratio must look like 2:1")
        return 2
    w = {"en": float(parts[0]), "zh": float(parts[1])}

    arrays = {}
    for key in w:
        p = resolve(f"data/processed/src_{key}.bin")
        if not p.exists():
            print(f"ERROR: {p} missing; run `encode` first")
            return 2
        arrays[key] = np.memmap(p, dtype=dtype, mode="r")
        print(f"  {key}: {len(arrays[key]):,} tokens available")

    # ---- per-source budget ---------------------------------------------
    # Straight proportional split, clamped to what exists. Deliberately NO
    # redistribution: if one source runs short, the honest outcome is a smaller
    # total that still honours the ratio -- not a silently different ratio.
    total_w = sum(w.values())
    need = {}
    for k in w:
        want = int(args.target_tokens * w[k] / total_w)
        have = len(arrays[k])
        need[k] = min(want, have)
        if want > have:
            print(f"  [!!] {k}: wanted {want:,} but only {have:,} available")

    achieved = sum(need.values())
    print("\n  mix plan:")
    for k in w:
        print(f"    {k}: {need[k]:>13,} tokens  ({need[k]/achieved*100:4.1f}%)")
    note = "  (ratio preserved)" if achieved < args.target_tokens else ""
    print(f"    total: {achieved:>11,}  target was {args.target_tokens:,}{note}")

    # ---- train/val split decided UP FRONT ------------------------------
    # Deciding the boundary before writing avoids reopening a file that
    # np.memmap still has mapped. On Windows that fails with OSError 22
    # ("Invalid argument"), because the mapping holds the file open.
    val_n = min(args.val_tokens, achieved // 20)
    train_n = achieved - val_n
    out_train = resolve(args.out_train)
    out_val = resolve(args.out_val)
    out_train.parent.mkdir(parents=True, exist_ok=True)

    chunk = args.chunk_tokens
    pos = {k: 0 for k in w}
    written = 0
    t0 = time.time()

    ftrain = out_train.open("wb")
    fval = out_val.open("wb")
    try:
        while True:
            progressed = False
            for k in w:
                if pos[k] >= need[k]:
                    continue
                take = min(chunk, need[k] - pos[k])
                block = np.asarray(arrays[k][pos[k] : pos[k] + take])
                pos[k] += take
                progressed = True

                # The train/val boundary can fall in the middle of a block.
                off = 0
                while off < len(block):
                    if written < train_n:
                        n = min(len(block) - off, train_n - written)
                        ftrain.write(block[off : off + n].tobytes())
                    else:
                        n = len(block) - off
                        fval.write(block[off : off + n].tobytes())
                    written += n
                    off += n
            if not progressed:
                break
    finally:
        ftrain.close()
        fval.close()

    print(f"\n  wrote {written:,} tokens in {time.time()-t0:.0f}s")
    print(f"    train {train_n:>13,}  {out_train.stat().st_size/1e6:7.0f} MB  -> {out_train.name}")
    print(f"    val   {val_n:>13,}  {out_val.stat().st_size/1e6:7.0f} MB  -> {out_val.name}")

    meta = {
        "tokenizer": str(resolve(args.tokenizer)),
        "vocab_size": tok.vocab_size,
        "dtype": str(dtype),
        "ratio": args.ratio,
        "per_source_tokens": need,
        "train_tokens": train_n,
        "val_tokens": val_n,
        "chunk_tokens": chunk,
    }
    meta_path = out_train.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- verify the mix by DECODING actual windows ----------------------
    # Checking that the ratio is right by construction is not enough: an
    # interleaving bug produces a file of the right size with the wrong content.
    # So read real windows back and decode them.
    print("\n  verifying by decoding windows from train.bin:")
    data = np.memmap(out_train, dtype=dtype, mode="r")
    n = len(data)
    for frac, label in ((0.00, "start"), (0.25, "25%"), (0.50, "50%"), (0.90, "90%")):
        s = int(n * frac)
        window = np.asarray(data[s : s + 160], dtype=np.int64).tolist()
        text = tok.decode(window, skip_special_tokens=True)
        text = " ".join(text.split())
        print(f"    [{label:>5}] {text[:96]}")
    del data
    print(f"\n  meta: {meta_path}")
    return 0


# --------------------------------------------------------------------------


def build_common() -> argparse.ArgumentParser:
    """
    Options shared by every subcommand.

    Uses a parent parser, NOT options on the top-level parser, so both
    `build_corpus.py --preset base encode` and `build_corpus.py encode --preset base`
    work. With top-level-only options the second form is rejected with
    "unrecognized arguments", which is the form everyone actually types.
    """
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--preset", default="base", choices=sorted(PRESETS))
    c.add_argument("--tokenizer", default="data/tokenizer/base_tokenizer.json")
    c.add_argument("--sample-mb", type=int, default=64,
                   help="per-source byte sample used to train the vocabulary. "
                        "BPE merge statistics converge fast; 64 MB is plenty for "
                        "a 32k vocabulary and keeps training to well under a minute.")
    c.add_argument("--batch-docs", type=int, default=20000)
    c.add_argument("--target-tokens", type=int, default=630_000_000,
                   help="Chinchilla optimum for the base preset is roughly 630M")
    c.add_argument("--ratio", default="2:1", help="en:zh")
    c.add_argument("--chunk-tokens", type=int, default=65_536,
                   help="granularity of the en/zh interleaving")
    c.add_argument("--val-tokens", type=int, default=2_000_000)
    c.add_argument("--out-train", default="data/processed/train.bin")
    c.add_argument("--out-val", default="data/processed/val.bin")
    c.add_argument("--only", nargs="*", choices=sorted(SOURCES),
                   help="restrict to these sources")
    c.add_argument("--force", action="store_true", help="redo work that already exists")
    return c


def main() -> int:
    common = build_common()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (
        ("extract", "parquet -> plain text, one file per source"),
        ("tokenizer", "train one BPE vocabulary on a sample of all sources"),
        ("encode", "tokenise each source into its own .bin"),
        ("mix", "interleave the .bins at a target ratio"),
        ("all", "run every stage in order"),
    ):
        sub.add_parser(name, parents=[common], help=helptext, description=helptext)

    args = ap.parse_args()

    if args.cmd == "extract":
        return cmd_extract(args)
    if args.cmd == "tokenizer":
        return cmd_tokenizer(args)
    if args.cmd == "encode":
        return cmd_encode(args)
    if args.cmd == "mix":
        return cmd_mix(args)
    if args.cmd == "all":
        for fn in (cmd_extract, cmd_tokenizer, cmd_encode, cmd_mix):
            rc = fn(args)
            if rc:
                return rc
            print()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
