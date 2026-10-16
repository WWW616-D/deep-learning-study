"""
Pretraining data: tokenise once into a flat binary, then memory-map it.

Why a flat .bin instead of a list of token lists
------------------------------------------------
The TinyStories corpus is ~1.9 GB of text, about 480M tokens. Holding that as a
Python list of ints would need tens of gigabytes of RAM. Written as a flat
uint16 array it is 960 MB on disk, and np.memmap lets the OS page it in on
demand, so the training process uses almost no RAM regardless of corpus size.

Why uint16
----------
Our vocabulary is 32768, which fits in uint16 (max 65535). That halves the file
size versus uint32 and roughly halves the disk traffic during memmap reads.
dtype_for_vocab() raises if a vocabulary ever gets too big for the chosen width,
so this optimisation can never silently truncate token ids.

Document boundaries
-------------------
Each document is wrapped as BOS ... EOS. Without a terminator the model has no
way to learn where one story ends and the next begins, and generation drifts
into a continuous stream with no stopping point. The cost is 2 tokens per
document -- negligible against ~250 tokens per story.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.template import BOS_ID, EOS_ID

# --------------------------------------------------------------------------
#  dtypes
# --------------------------------------------------------------------------


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    if vocab_size <= 2**16:
        return np.dtype(np.uint16)
    if vocab_size <= 2**32:
        return np.dtype(np.uint32)
    raise ValueError(f"vocabulary of {vocab_size} does not fit in uint32")


# --------------------------------------------------------------------------
#  Preparing the binary
# --------------------------------------------------------------------------


DEFAULT_DOC_SEPARATOR = "<|endoftext|>"


def _iter_separated(fh, separator: str, chunk_size: int = 1 << 20) -> Iterator[str]:
    """
    Split a text stream on `separator` without loading the whole file.

    The separator can straddle a chunk boundary, so the trailing partial piece is
    carried into the next chunk rather than emitted as a short fragment.
    """
    tail = ""
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            break
        parts = (tail + chunk).split(separator)
        tail = parts.pop()
        yield from parts
    if tail:
        yield tail


def iter_documents(
    paths: Sequence[str | Path],
    max_docs: int | None = None,
    separator: str | None = DEFAULT_DOC_SEPARATOR,
) -> Iterator[str]:
    """
    Yield one whole document at a time.

    IMPORTANT: TinyStories-train.txt is NOT one story per line. Each story is
    wrapped across several lines and stories are separated by a literal
    <|endoftext|> marker. Treating a line as a document silently trains on
    sentence fragments: it produced 35-token "stories" instead of ~200-token
    ones, understating the corpus by roughly 6x, and it would have taught the
    model that stories end at arbitrary points mid-sentence.

    Whitespace inside a document is normalised to single spaces, because the
    line breaks in the source file are formatting, not content.

    Pass separator=None for corpora that really are newline-delimited.
    """
    seen = 0
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            source: Iterable[str] = _iter_separated(fh, separator) if separator else fh
            for raw in source:
                doc = " ".join(raw.split())
                if not doc:
                    continue
                yield doc
                seen += 1
                if max_docs is not None and seen >= max_docs:
                    return


def prepare_bin(
    tok,
    paths: Sequence[str | Path],
    out_path: str | Path,
    *,
    wrap_docs: bool = True,
    batch_docs: int = 2000,
    max_docs: int | None = None,
    separator: str | None = DEFAULT_DOC_SEPARATOR,
    log_every: int = 50_000,
    verbose: bool = True,
) -> dict:
    """
    Tokenise documents into a flat binary file.

    Returns a small report including the token count and the compression ratio,
    which is the number you want to look at: if tokens/char is far above what
    you expected, the vocabulary is too small for the data's language mix.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = dtype_for_vocab(tok.vocab_size)
    n_tokens = 0
    n_docs = 0
    n_chars = 0

    with out_path.open("wb") as fh:
        buffer: List[str] = []

        def flush() -> None:
            nonlocal n_tokens, n_chars, n_docs
            if not buffer:
                return
            ids_batch = tok.encode_batch(buffer)
            for ids in ids_batch:
                if wrap_docs:
                    seq = [BOS_ID, *ids, EOS_ID]
                else:
                    seq = ids
                arr = np.asarray(seq, dtype=dtype)
                fh.write(arr.tobytes())
                n_tokens += len(seq)
            n_chars += sum(len(b) for b in buffer)
            n_docs += len(buffer)
            buffer.clear()

        for doc in iter_documents(paths, max_docs=max_docs, separator=separator):
            buffer.append(doc)
            if len(buffer) >= batch_docs:
                flush()
                if verbose and n_docs % log_every < batch_docs:
                    print(f"    {n_docs:>9,} docs  {n_tokens:>12,} tokens", flush=True)
        flush()

    report = {
        "path": str(out_path),
        "documents": n_docs,
        "tokens": n_tokens,
        "chars": n_chars,
        "tokens_per_char": n_tokens / n_chars if n_chars else 0.0,
        "bytes_on_disk": out_path.stat().st_size,
        "dtype": str(dtype),
    }
    if verbose:
        print(
            f"  wrote {out_path.name}: {n_docs:,} docs, {n_tokens:,} tokens, "
            f"{report['bytes_on_disk']/1e6:.1f} MB, {report['tokens_per_char']:.3f} tokens/char"
        )
    return report


def split_bin(
    src: str | Path,
    train_path: str | Path,
    val_path: str | Path,
    *,
    val_tokens: int,
    dtype: np.dtype = np.dtype(np.uint16),
) -> Tuple[int, int]:
    """
    Split a flat token file into train/val by TOKENS, not by documents.

    Splitting by tokens guarantees the validation set has exactly the size you
    asked for and that no token appears in both halves.
    """
    src = Path(src)
    data = np.memmap(src, dtype=dtype, mode="r")
    total = len(data)
    if val_tokens >= total:
        raise ValueError(f"val_tokens={val_tokens} >= total tokens {total}")

    train_n = total - val_tokens
    train_path, val_path = Path(train_path), Path(val_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("wb") as f:
        f.write(np.asarray(data[:train_n]).tobytes())
    with val_path.open("wb") as f:
        f.write(np.asarray(data[train_n:]).tobytes())
    return train_n, val_tokens


# --------------------------------------------------------------------------
#  Reading during training
# --------------------------------------------------------------------------


class TokenizedDataset(Dataset):
    """
    Memory-mapped next-token-prediction dataset.

    Index i maps to the window [i*seq_len, i*seq_len + seq_len + 1):
        x = window[:-1]   the inputs
        y = window[1:]    the targets, i.e. x shifted left by one

    The shift is the whole point: position t must predict token t+1. Doing it
    here (rather than in the model) keeps the model a pure function of x, which
    makes the KV-cache equivalence test meaningful.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        dtype: np.dtype = np.dtype(np.uint16),
    ) -> None:
        self.path = str(path)
        self.seq_len = int(seq_len)
        self.data = np.memmap(path, dtype=dtype, mode="r")
        if len(self.data) < self.seq_len + 1:
            raise ValueError(
                f"{path} holds {len(self.data)} tokens, need at least {self.seq_len + 1}"
            )

    def __len__(self) -> int:
        # Whole windows only; a trailing partial window is dropped rather than
        # zero-padded, which would train the model to predict padding.
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i: int):
        start = i * self.seq_len
        chunk = np.asarray(self.data[start : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])

    @property
    def n_tokens(self) -> int:
        return len(self.data)


class TokenSlice(TokenizedDataset):
    """
    A view over a contiguous token range, sharing the parent's mmap.

    Used to carve a validation set out of train.bin when no separate val.bin
    exists. Slicing by TOKENS (not documents) guarantees the two ranges are
    disjoint and that validation has exactly the size asked for, with no data
    copied and no extra memory.
    """

    def __init__(self, base: TokenizedDataset, start: int, length: int) -> None:
        if start < 0 or length <= 0 or start + length > len(base.data):
            raise ValueError(
                f"slice [{start}, {start + length}) is outside a {len(base.data)}-token file"
            )
        self.path = base.path + f"[{start}:{start + length}]"
        self.seq_len = base.seq_len
        self.data = base.data[start : start + length]
        if length < self.seq_len + 1:
            raise ValueError(f"slice of {length} tokens is shorter than seq_len+1")


def make_dataloader(
    path: str | Path,
    seq_len: int,
    batch_size: int,
    *,
    dtype: np.dtype = np.dtype(np.uint16),
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 0,
) -> DataLoader:
    ds = TokenizedDataset(path, seq_len, dtype)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        generator=gen if shuffle else None,
        persistent_workers=num_workers > 0,
    )


def bin_stats(path: str | Path, dtype: np.dtype = np.dtype(np.uint16)) -> dict:
    data = np.memmap(path, dtype=dtype, mode="r")
    n = len(data)
    if n == 0:
        return {"tokens": 0}
    sample = np.asarray(data[: min(n, 2_000_000)])
    counts = np.bincount(sample, minlength=int(sample.max()) + 1)
    # Fraction of tokens that are the most common one: a crude health check.
    top = counts.max() / counts.sum() if counts.sum() else 0.0
    return {
        "tokens": int(n),
        "distinct_in_sample": int((counts > 0).sum()),
        "most_common_fraction": float(top),
        "mean_token_id": float(sample.mean()),
    }


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(p, json.dumps(bin_stats(p), indent=2))
