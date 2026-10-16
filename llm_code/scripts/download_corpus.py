#!/usr/bin/env python
"""
Download pretraining corpora through hf-mirror.

Sources are declared in a registry so adding one is a two-line change. Each is
a HuggingFace *datasets* repo whose data lives at
`<mirror>/datasets/<repo>/resolve/main/<path>`.

Usage
-----
    # see what is available
    python scripts/download_corpus.py --list

    # the recommended starting set: one English parquet + all Chinese wiki
    python scripts/download_corpus.py --source fineweb-edu-en --limit 1
    python scripts/download_corpus.py --source wikipedia-zh

    # sizes without downloading
    python scripts/download_corpus.py --source finweb-edu-en --limit 2 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.net import download, human, remote_size  # noqa: E402

MIRROR = "https://hf-mirror.com"


def _tree(repo: str, sub: str = "") -> list[dict]:
    """List files under a datasets-repo subdirectory, via the HF API."""
    import json
    import urllib.request

    path = f"/api/datasets/{repo}/tree/main" + (f"/{sub}" if sub else "")
    req = urllib.request.Request(MIRROR + path, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as r:
        entries = json.load(r)
    return [e for e in entries if e.get("type") == "file"]


SOURCES: dict[str, dict] = {
    "fineweb-edu-en": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "sub": "sample/10BT",
        "out": "data/raw/fineweb-edu",
        "columns": ["text"],
        "note": "Educational web text, quality-filtered. ~715M tokens per 2 GB file. "
                "The one corpus SmolLM2 also used at small scale.",
    },
    "wikipedia-zh": {
        "repo": "wikimedia/wikipedia",
        "sub": "20231101.zh",
        "out": "data/raw/wikipedia-zh",
        "columns": ["text"],
        "note": "Chinese Wikipedia. Encyclopedic: long sentences, large vocabulary. "
                "HARD for a small model -- expect weaker Chinese than English.",
    },
    "wikipedia-en": {
        "repo": "wikimedia/wikipedia",
        "sub": "20231101.en",
        "out": "data/raw/wikipedia-en",
        "columns": ["text"],
        "note": "English Wikipedia, 10.8 GB total across 41 files.",
    },
    "cosmopedia-v2": {
        "repo": "HuggingFaceTB/smollm-corpus",
        "sub": "cosmopedia-v2",
        "out": "data/raw/cosmopedia-v2",
        "columns": ["text"],
        "note": "Synthetic textbooks (Mixtral-generated). 114 GB. High quality but large.",
    },
}


def list_sources() -> None:
    print(f"mirror: {MIRROR}\n")
    for name, s in SOURCES.items():
        print(f"  {name}")
        print(f"      repo : {s['repo']}  sub={s['sub']}")
        print(f"      note : {s['note']}")
        try:
            files = _tree(s["repo"], s["sub"])
            total = sum(f.get("size", 0) for f in files)
            print(f"      files: {len(files)}  total {human(total)}")
            for f in files[:3]:
                print(f"             {Path(f['path']).name:<44} {human(f.get('size', 0))}")
            if len(files) > 3:
                print(f"             ... and {len(files)-3} more")
        except Exception as e:
            print(f"      files: (listing failed: {type(e).__name__}: {str(e)[:50]})")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", choices=sorted(SOURCES))
    ap.add_argument("--limit", type=int, default=0, help="0 = all files")
    ap.add_argument("--out-dir", default=None, help="override the registry's output dir")
    ap.add_argument("--mirror", default=MIRROR)
    ap.add_argument("--list", action="store_true", help="show available sources and sizes")
    ap.add_argument("--dry-run", action="store_true", help="HEAD each file, download nothing")
    args = ap.parse_args()

    if args.list or not args.source:
        list_sources()
        return 0

    s = SOURCES[args.source]
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / s["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"source   : {args.source}  ({s['repo']} / {s['sub']})")
    print(f"out dir  : {out_dir}")
    print(f"note     : {s['note']}")
    print("-" * 70)

    files = _tree(s["repo"], s["sub"])
    if args.limit:
        files = files[: args.limit]

    total_bytes = 0
    for i, f in enumerate(files, 1):
        name = Path(f["path"]).name
        dest = out_dir / name
        url = f"{args.mirror}/datasets/{s['repo']}/resolve/main/{f['path']}"
        size = f.get("size") or remote_size(url)

        if dest.exists():
            print(f"[{i}/{len(files)}] {name}: already present ({human(dest.stat().st_size)})")
            total_bytes += dest.stat().st_size
            continue
        if args.dry_run:
            print(f"[{i}/{len(files)}] {name}: would download {human(size) if size else '?'}")
            continue

        print(f"[{i}/{len(files)}] {name}  ({human(size) if size else '?'})")
        try:
            got = download(url, dest)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            continue
        total_bytes += got
        print(f"    done: {human(got)}\n")

    print("-" * 70)
    print(f"total present: {human(total_bytes)}")
    print(f"\nnext: python scripts/build_corpus.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
