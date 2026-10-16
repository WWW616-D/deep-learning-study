#!/usr/bin/env python
"""
Download the TinyStories corpus through a HuggingFace mirror.

Why a custom downloader rather than huggingface_hub
---------------------------------------------------
huggingface.co is unreachable from this machine (the host resolves to 127.0.0.1
via an accelerator's hosts entry, and direct TLS fails), so everything goes
through hf-mirror.com. The corpus is two plain text files, so a resumable
streaming download is simpler and more debuggable than pulling the whole
`datasets` machinery in.

1.92 GB over a slow link will be interrupted at least once, so the download
writes to `<name>.part` and resumes with an HTTP Range request. Nothing is
renamed into place until the byte count matches what the server reported.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MIRROR = "https://hf-mirror.com"
REPO = "datasets/roneneldan/TinyStories/resolve/main"

FILES = {
    "train": "TinyStories-train.txt",
    "valid": "TinyStories-valid.txt",
    # V2 is GPT-4 generated and cleaner, but the published TinyStories-1M/3M/8M/28M
    # baselines were trained on V1 -- use V1 if you want to compare against them.
    "train_v2": "TinyStoriesV2-GPT4-train.txt",
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, *, quiet_every: float = 5.0) -> int:
    """Stream `url` to `dest`, resuming a partial `.part` file if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    pos = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (resumable-downloader)"}
    if pos:
        headers["Range"] = f"bytes={pos}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and pos:  # already complete
            part.rename(dest)
            return dest.stat().st_size
        raise

    with resp:
        if resp.status == 206:
            content_range = resp.headers.get("Content-Range", "")
            total = int(content_range.split("/")[-1]) if "/" in content_range else None
            if pos:
                print(f"  resuming at {human(pos)}")
        else:
            pos = 0
            total = int(resp.headers["Content-Length"]) if resp.headers.get("Content-Length") else None
            part.unlink(missing_ok=True)

        mode = "ab" if pos else "wb"
        written = pos
        t0 = time.time()
        last = 0.0
        with part.open(mode) as fh:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last >= quiet_every:
                    last = now
                    speed = (written - pos) / max(now - t0, 1e-6)
                    if total:
                        pct = written / total * 100
                        eta = (total - written) / max(speed, 1)
                        print(
                            f"    {human(written)} / {human(total)}  {pct:5.1f}%  "
                            f"{human(speed)}/s  eta {eta/60:.1f} min",
                            flush=True,
                        )
                    else:
                        print(f"    {human(written)}  {human(speed)}/s", flush=True)

    size = part.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(
            f"incomplete download: got {size} bytes, server said {total}. "
            f"Keeping {part.name}; rerun to resume."
        )
    part.rename(dest)
    return size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/tinystories")
    ap.add_argument("--what", nargs="*", default=["train", "valid"],
                    choices=sorted(FILES) + ["all"])
    ap.add_argument("--mirror", default=MIRROR)
    args = ap.parse_args()

    # Keep this script dependency-free (it runs before the venv even matters),
    # so resolve against ROOT inline rather than importing src.paths.
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    what = sorted(FILES) if "all" in args.what else args.what

    free = shutil.disk_usage(out_dir.parent if out_dir.parent.exists() else ROOT).free
    print(f"mirror   : {args.mirror}")
    print(f"out dir  : {out_dir}")
    print(f"free disk: {human(free)}")
    print("-" * 68)

    total_bytes = 0
    for key in what:
        name = FILES[key]
        url = f"{args.mirror}/{REPO}/{name}"
        dest = out_dir / name
        if dest.exists():
            size = dest.stat().st_size
            print(f"[{key}] already present: {name}  {human(size)}")
            total_bytes += size
            continue
        print(f"[{key}] {url}")
        try:
            size = download(url, dest)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        total_bytes += size
        print(f"  done: {human(size)}")
        print()

    print("-" * 68)
    print(f"total downloaded/present: {human(total_bytes)}")
    print("\nnext: convert to a tokenised .bin with scripts/prepare_pretrain_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
