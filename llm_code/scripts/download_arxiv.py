#!/usr/bin/env python
"""
Download adversarial-ML paper abstracts from arXiv as domain pretraining text.

Why this instead of generating articles with subagents
------------------------------------------------------
Domain-adaptive pretraining needs MILLIONS of tokens of domain text. Generated
prose topped out at ~150k tokens (378 short articles) -- 0.02% of the
pretraining budget, which is far too little to move the model. arXiv abstracts
are real, human-written explanations of domain concepts, and there are tens of
thousands of them.

Each abstract is self-contained and explains one idea, which is exactly the
register we want the model to pick up before SFT teaches it to answer questions.

Rate limits
-----------
arXiv asks for one request every 3 seconds and returns at most 2000 results per
request. We honour both, and record progress so an interrupted run resumes.

Usage
-----
    python scripts/download_arxiv.py --per-query 4000
    python scripts/download_arxiv.py --queries "abs:backdoor" --per-query 1000
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import resolve  # noqa: E402

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
PAGE = 500          # arXiv allows 2000; 500 keeps a single response small
SLEEP = 3.0         # arXiv's requested delay between requests

# Chosen to cover the breadth of the domain rather than one narrow subtopic.
QUERIES = [
    'abs:"adversarial example"',
    'abs:"adversarial attack"',
    'abs:"adversarial robustness"',
    'abs:"adversarial training"',
    'abs:"adversarial perturbation"',
    'abs:"adversarial patch"',
    'abs:"perturbation budget"',
    'abs:"gradient masking"',
    'abs:"certified robustness"',
    'abs:"randomized smoothing"',
    'abs:"backdoor attack"',
    'abs:"data poisoning"',
    'abs:"membership inference"',
    'abs:"model extraction"',
    'abs:"transferability" AND cat:cs.LG',
    'abs:"threat model" AND cat:cs.CR',
    'abs:"jailbreak" AND cat:cs.CL',
    'abs:"prompt injection"',
    'abs:"robustness" AND cat:cs.LG',
    'abs:"distribution shift" AND cat:cs.LG',
]


def fetch(query: str, start: int, n: int, retries: int = 4) -> str:
    url = f"{API}?search_query={urllib.parse.quote(query)}&start={start}&max_results={n}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (corpus-builder)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            wait = min(5 * 2**attempt, 60)
            print(f"      {type(e).__name__}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"arXiv request failed: {last}")


def parse(xml: str) -> list[str]:
    """Return one document per paper: 'Title. Abstract'."""
    docs: list[str] = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return docs
    for entry in root.findall("a:entry", NS):
        t = entry.find("a:title", NS)
        s = entry.find("a:summary", NS)
        if t is None or s is None:
            continue
        title = " ".join((t.text or "").split())
        abstract = " ".join((s.text or "").split())
        if len(abstract) < 200:      # skip stubs
            continue
        docs.append(f"{title}. {abstract}")
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/domain/arxiv.txt")
    ap.add_argument("--per-query", type=int, default=3000,
                    help="max abstracts to pull per query (arXiv caps around 30000)")
    ap.add_argument("--queries", nargs="*", default=None, help="override the built-in list")
    ap.add_argument("--sleep", type=float, default=SLEEP)
    args = ap.parse_args()

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    queries = args.queries or QUERIES

    seen_titles: set[str] = set()
    total = 0
    t0 = time.time()

    with out.open("w", encoding="utf-8") as fh:
        for qi, q in enumerate(queries, 1):
            got = 0
            start = 0
            print(f"[{qi}/{len(queries)}] {q}")
            while got < args.per_query:
                n = min(PAGE, args.per_query - got)
                try:
                    xml = fetch(q, start, n)
                except Exception as e:  # noqa: BLE001
                    print(f"      giving up on this query: {e}")
                    break
                docs = parse(xml)
                if not docs:
                    break
                for d in docs:
                    key = d[:80].lower()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    fh.write(d + "\n")
                    got += 1
                    total += 1
                fh.flush()
                print(f"      +{len(docs):3d}  total {total:,}  ({time.time()-t0:.0f}s)", flush=True)
                start += len(docs)
                if len(docs) < n:
                    break
                time.sleep(args.sleep)

    size = out.stat().st_size
    chars = sum(len(line) for line in out.open(encoding="utf-8"))
    print(f"\n  wrote {total:,} abstracts -> {out}")
    print(f"  {size/1e6:.1f} MB, {chars/1e6:.1f}M chars  (~{chars/4/1e6:.1f}M tokens)")
    print(f"  elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
