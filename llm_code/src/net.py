"""
Resumable HTTP downloads.

Shared by scripts/download_tinystories.py and scripts/download_corpus.py.

Why hand-rolled rather than huggingface_hub
-------------------------------------------
huggingface.co is unreachable from this machine (the host resolves to 127.0.0.1
via an accelerator's hosts entry, and direct TLS fails), so everything goes
through hf-mirror.com. The corpora we need are plain files behind a
`/resolve/main/...` URL, so a resumable streaming download is simpler and far
easier to debug than pulling in the whole `datasets` stack.

A multi-gigabyte download over a ~10 MB/s link WILL be interrupted at least
once, so the download writes to `<name>.part` and resumes with an HTTP Range
request. The file is only renamed into place once the byte count matches what
the server reported.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20  # 1 MB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download(
    url: str,
    dest: str | Path,
    *,
    quiet_every: float = 5.0,
    verbose: bool = True,
) -> int:
    """Stream `url` into `dest`, resuming a partial `.part` file if present."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    pos = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (resumable-downloader)"}
    if pos:
        headers["Range"] = f"bytes={pos}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=90)
    except urllib.error.HTTPError as e:
        if e.code == 416 and pos:
            # Server says the range is unsatisfiable: the file is already whole.
            part.replace(dest)
            return dest.stat().st_size
        raise

    with resp:
        if resp.status == 206:
            content_range = resp.headers.get("Content-Range", "")
            total = int(content_range.split("/")[-1]) if "/" in content_range else None
            if pos and verbose:
                print(f"    resuming at {human(pos)}")
        else:
            pos = 0
            cl = resp.headers.get("Content-Length")
            total = int(cl) if cl else None
            part.unlink(missing_ok=True)

        mode = "ab" if pos else "wb"
        written = pos
        t0 = time.time()
        last = 0.0
        with part.open(mode) as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
                now = time.time()
                if verbose and now - last >= quiet_every:
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
            f"incomplete download: got {size} bytes, server reported {total}. "
            f"Keeping {part.name}; rerun to resume."
        )
    part.replace(dest)
    return size


def remote_size(url: str) -> int | None:
    """Content-Length of a URL, or None if the server will not say."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None
