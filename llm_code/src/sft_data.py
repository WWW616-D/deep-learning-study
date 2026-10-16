"""
SFT data pipeline: conversations in, (token ids, loss mask) out.

The one thing that must not go wrong
------------------------------------
The token sequence we train on MUST be byte-identical to what src/template.py
produces at inference time. A single extra space, or a role token encoded
slightly differently, and the model sees an input distribution it never trained
on. The failure looks like "the model is just bad", not like a bug.

So build_example() does NOT hand-roll the string. It assembles the sequence
role-token by role-token, and tests/test_core.py asserts the result equals
`tok.encode(build_prompt(...))` exactly. That assertion is the whole safety net.

Why the mask is shifted
-----------------------
At position t the model predicts token t+1. So a mask entry answering "should
we supervise the token at index i?" has to be shifted the same way the targets
are:

    x    = ids[:-1]
    y    = ids[1:]
    mask = turn_mask[1:]     <-- shifted, so mask[t] supervises ids[t+1]

Getting this off by one silently trains the model on the wrong tokens, and the
loss still goes down, so it is easy to miss.

What is supervised
------------------
    <|bos|>            no
    <|system|>         no
    system content     no      -- the model does not need to learn to say it
    <|eos|>            no
    <|user|>           no
    user content       no      -- this is given; predicting it wastes capacity
    <|eos|>            no
    <|assistant|>      no      -- supplied as a prompt at inference
    assistant content  YES
    <|eos|>            YES     -- so the model learns to STOP
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.template import (
    BOS_ID,
    DEFAULT_SYSTEM,
    EOS_ID,
    ROLE_TO_ID,
    SYSTEM_ID,
    Message,
)


# --------------------------------------------------------------------------
#  Building one example
# --------------------------------------------------------------------------


def build_example(
    tok,
    messages: Sequence[Message],
    system: str | None = None,
    *,
    supervise_system: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Turn one conversation into (ids, turn_mask).

    turn_mask[i] == 1 means: supervise the prediction of ids[i] when it plays
    the role of a target (i.e. after the standard shift).
    """
    if system is None:
        system = DEFAULT_SYSTEM
    system = system.strip()

    ids: List[int] = [BOS_ID]
    mask: List[int] = [0]

    # ---- system turn -------------------------------------------------
    s = 1 if supervise_system else 0
    ids.append(SYSTEM_ID)
    mask.append(0)
    if system:
        piece = tok.encode(system)
        ids.extend(piece)
        mask.extend([s] * len(piece))
    ids.append(EOS_ID)
    mask.append(s)

    # ---- conversation turns ------------------------------------------
    for msg in messages:
        role = msg["role"]
        if role not in ROLE_TO_ID:
            raise ValueError(f"unsupported role {role!r} in SFT data")
        m = 1 if role == "assistant" else 0

        # Role tokens are supplied as input at inference, never predicted.
        ids.append(ROLE_TO_ID[role])
        mask.append(0)

        content = msg["content"].strip()
        if content:
            piece = tok.encode(content)
            ids.extend(piece)
            mask.extend([m] * len(piece))

        # The EOS that CLOSES an assistant turn is supervised: that is how the
        # model learns to stop talking. The EOS closing a user turn is not.
        ids.append(EOS_ID)
        mask.append(m)

    return ids, mask


# --------------------------------------------------------------------------
#  Loading jsonl
# --------------------------------------------------------------------------


def load_jsonl(paths: Iterable[str | Path]) -> List[dict]:
    out: List[dict] = []
    for p in paths:
        p = Path(p)
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e
                msgs = obj.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    raise ValueError(f"{p}:{lineno}: missing 'messages'")
                out.append(obj)
    return out


def dedupe_by_question(samples: List[dict]) -> List[dict]:
    """
    Drop repeated first-user-questions, keeping the first occurrence.

    Repeating one question with different answers teaches the model that the
    answer is arbitrary, which is worse than having less data.
    """
    seen = set()
    kept = []
    for s in samples:
        q = next((m["content"] for m in s["messages"] if m["role"] == "user"), None)
        key = " ".join(q.strip().lower().split()) if q else None
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)
    return kept


# --------------------------------------------------------------------------
#  Dataset
# --------------------------------------------------------------------------


class SFTDataset(Dataset):
    """
    Fixed-length padded batches.

    Padding is acceptable here: SFT sets are small (hundreds to tens of
    thousands of examples) and conversations are short, so the wasted compute is
    minor. For a large set, pack multiple conversations into one sequence
    instead -- but then attention leaks across conversation boundaries and you
    need a block-diagonal mask to prevent it.
    """

    def __init__(
        self,
        samples: Sequence[dict],
        tok,
        max_len: int,
        *,
        pad_id: int = 0,
        verbose: bool = True,
    ) -> None:
        self.max_len = max_len
        self.pad_id = pad_id
        self.ids: List[List[int]] = []
        self.masks: List[List[int]] = []
        self.skipped_no_supervision = 0
        self.truncated = 0
        self.turn_pairs = 0

        for s in samples:
            ids, mask = build_example(tok, s["messages"], s.get("system"))
            if sum(mask) == 0:
                # Nothing to learn from: an all-zero mask makes the loss a
                # constant zero, which quietly wastes a training step.
                self.skipped_no_supervision += 1
                continue
            if len(ids) > max_len:
                self.truncated += 1
                ids = ids[:max_len]
                mask = mask[:max_len]

            pad = max_len - len(ids)
            self.ids.append(ids + [pad_id] * pad)
            # Padding is NEVER supervised: predicting pad tokens would teach the
            # model to emit them.
            self.masks.append(mask + [0] * pad)
            self.turn_pairs += len(s["messages"]) // 2

        if verbose:
            n = len(self.ids)
            lens = [sum(1 for _ in self.ids[i]) - self.ids[i].count(pad_id) for i in range(n)]
            sup = [sum(m) for m in self.masks]
            print(f"    examples          : {n}")
            print(f"    turn pairs total  : {self.turn_pairs}  (avg {self.turn_pairs/max(n,1):.2f}/example)")
            print(f"    real length       : min {min(lens)}  med {sorted(lens)[n//2]}  max {max(lens)}")
            print(f"    supervised tokens : avg {sum(sup)/max(n,1):.1f} / {max_len}")
            print(f"    supervised share  : {sum(sup)/max(n*max_len,1)*100:.1f}%")
            if self.truncated:
                print(f"    truncated         : {self.truncated}")
            if self.skipped_no_supervision:
                print(f"    skipped (no sup.) : {self.skipped_no_supervision}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        ids = torch.tensor(self.ids[i], dtype=torch.long)
        mask = torch.tensor(self.masks[i], dtype=torch.float32)
        # Same shift as every other dataset in this project:
        # position t predicts token t+1, so the mask is shifted to match.
        return ids[:-1], ids[1:], mask[1:]

    @property
    def n_supervised_tokens(self) -> int:
        return sum(sum(m) for m in self.masks)


def split_samples(
    samples: List[dict], val_fraction: float, seed: int = 0
) -> Tuple[List[dict], List[dict]]:
    """Deterministic shuffle-then-split so reruns put the same items in val."""
    import random

    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(samples) * val_fraction)) if len(samples) > 1 else 0
    val_idx = set(idx[:n_val])
    train = [samples[i] for i in idx if i not in val_idx]
    val = [samples[i] for i in idx if i in val_idx]
    return train, val
