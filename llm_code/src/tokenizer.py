"""
Byte-level BPE tokenizer: training, saving, loading, encoding.

Why byte-level
--------------
A byte-level BPE has no unknown token: any UTF-8 string can be encoded, because
the alphabet starts as the 256 raw bytes. That matters for this project because
we are training on mixed English and Chinese, where a word-level vocabulary
would generate UNK on anything it had not seen.

Why the special tokens must come first
--------------------------------------
The trainer places `special_tokens` at the front of the vocabulary, so they take
ids 0, 1, 2, ... in the order given. src/template.py hard-codes PAD_ID=0 through
ASSISTANT_ID=5, so SPECIAL_TOKENS must stay in that exact order. verify() checks
this, because getting it wrong produces a tokenizer that silently mislabels every
role token -- a bug that is very hard to see in the loss curve.

Why add_prefix_space=False
--------------------------
That flag is for encoding isolated words. We always encode whole documents, so
the spaces are already present in the text and adding another one would corrupt
the first token of every document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from src.template import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    SYSTEM_ID,
    USER_ID,
)

DEFAULT_VOCAB_SIZE = 32768


def _byte_level_tokenizer() -> _HFTokenizer:
    tok = _HFTokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    corpus: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    *,
    show_progress: bool = True,
    min_frequency: int = 2,
) -> _HFTokenizer:
    """
    Train a byte-level BPE on an iterable of raw strings.

    The iterable may be a generator -- for a 2 GB corpus we never want the whole
    thing in memory at once, so it is consumed lazily by the Rust trainer.
    """
    tok = _byte_level_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=min_frequency,
        show_progress=show_progress,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


def train_from_files(
    paths: Sequence[Path | str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    *,
    show_progress: bool = True,
    max_bytes: int | None = None,
) -> _HFTokenizer:
    """
    Train from text files, optionally reading only the first `max_bytes`.

    Sampling is the normal practice: BPE merge statistics converge long before
    you have read a multi-gigabyte corpus, and training on ~200 MB instead of
    2 GB turns a 20-minute job into a 2-minute one for no measurable loss.
    """
    paths = [Path(p) for p in paths]

    def _iter() -> Iterator[str]:
        remaining = max_bytes if max_bytes is not None else -1
        for path in paths:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                while True:
                    if remaining >= 0 and remaining == 0:
                        return
                    chunk = f.read(1_000_000)
                    if not chunk:
                        break
                    if remaining > 0:
                        chunk = chunk[:remaining]
                        remaining -= len(chunk)
                    yield chunk
            if remaining == 0:
                return

    return train_tokenizer(_iter(), vocab_size=vocab_size, show_progress=show_progress)


# --------------------------------------------------------------------------
#  Wrapper
# --------------------------------------------------------------------------


class BPETokenizer:
    """Thin, explicit wrapper around the fast tokenizer."""

    def __init__(self, tok: _HFTokenizer) -> None:
        self._tok = tok
        self._special = set(SPECIAL_TOKENS)

    # ---- io ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(_HFTokenizer.from_file(str(path)))

    # ---- properties --------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def id_to_token(self, i: int) -> str | None:
        return self._tok.id_to_token(i)

    def token_to_id(self, t: str) -> int | None:
        return self._tok.token_to_id(t)

    def get_vocab(self) -> dict:
        return self._tok.get_vocab()

    # ---- encode / decode ---------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to ids.

        add_special_tokens=False by default: our conversations already carry an
        explicit BOS and every message carries an explicit EOS, so letting the
        library inject its own would duplicate them.
        """
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: Sequence[str], add_special_tokens: bool = False) -> List[List[int]]:
        return [
            e.ids
            for e in self._tok.encode_batch(list(texts), add_special_tokens=add_special_tokens)
        ]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # ---- verification ------------------------------------------------

    def verify(self, samples: Sequence[str] | None = None, verbose: bool = True) -> dict:
        """
        Assert the invariants the rest of the project depends on.

        Returns a report dict. Raises AssertionError on a contract violation,
        because a tokenizer that violates these is unusable, not merely suboptimal.
        """
        report: dict = {}

        # 1. special token ids must match template.py exactly
        expected = {
            "<|pad|>": PAD_ID,
            "<|bos|>": BOS_ID,
            "<|eos|>": EOS_ID,
            "<|system|>": SYSTEM_ID,
            "<|user|>": USER_ID,
            "<|assistant|>": ASSISTANT_ID,
        }
        got = {t: self.token_to_id(t) for t in expected}
        report["special_token_ids"] = got
        if got != expected:
            raise AssertionError(
                f"special token ids do not match src/template.py.\n"
                f"  expected {expected}\n  got      {got}"
            )

        # 2. no unknown token exists in the vocabulary
        report["has_unk"] = self.token_to_id("<unk>") is not None

        # 3. round-trip must be lossless for arbitrary unicode
        if samples is None:
            samples = [
                "Hello, world!",
                "Once upon a time there was a little cat.",
                "你好，世界。",
                "中英mixed文本 with punctuation!?",
                "emoji: \U0001f600 \U0001f680 and accents: café naïve",
                "  leading and trailing spaces  ",
                "\n\nnewlines\n\n",
                "0123456789 + - * / = < > %",
            ]
        bad: List[str] = []
        for s in samples:
            back = self.decode(self.encode(s))
            if back != s:
                bad.append(s)
        report["roundtrip_failures"] = bad
        if bad:
            raise AssertionError(f"round-trip is lossy for {len(bad)} sample(s): {bad[:3]!r}")

        # 4. compression ratio, reported per script
        def ratio(text: str) -> float:
            n = len(text)
            return len(self.encode(text)) / n if n else 0.0

        en = ("The quick brown fox jumps over the lazy dog. " * 20).strip()
        zh = ("这是一个用于测试分词压缩率的中文句子。" * 20).strip()
        report["tokens_per_char_en"] = ratio(en)
        report["tokens_per_char_zh"] = ratio(zh)

        # 5. role markers must survive as single tokens
        prompt = "<|bos|><|system|>sys<|eos|><|user|>hi<|eos|><|assistant|>"
        ids = self.encode(prompt)
        report["role_markers_are_atomic"] = ids == [
            BOS_ID, SYSTEM_ID, *self.encode("sys"), EOS_ID,
            USER_ID, *self.encode("hi"), EOS_ID, ASSISTANT_ID,
        ]

        if verbose:
            print(f"  vocab size            : {self.vocab_size:,}")
            print(f"  special token ids     : {got}")
            print(f"  unk token present     : {report['has_unk']}")
            print(f"  round-trip failures   : {len(bad)}")
            print(f"  tokens/char  english  : {report['tokens_per_char_en']:.3f}")
            print(f"  tokens/char  chinese  : {report['tokens_per_char_zh']:.3f}")
            print(f"  role markers atomic   : {report['role_markers_are_atomic']}")
            print(f"    -> ctx=1024 fits ~{int(1024 / report['tokens_per_char_en']):,} english chars, "
                  f"~{int(1024 / report['tokens_per_char_zh']):,} chinese chars")

        return report


def save_meta(tok: BPETokenizer, path: str | Path, **extra) -> Path:
    """Write a small JSON sidecar describing the tokenizer."""
    path = Path(path)
    meta = {
        "vocab_size": tok.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "ids": {
            "pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID,
            "system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID,
        },
    }
    meta.update(extra)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
