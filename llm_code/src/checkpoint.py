"""
Checkpointing that survives interruption.

Three kinds of checkpoint, because they answer three different questions:

  last.pt   full state (model + optimiser + step + RNG). Overwritten every save
            interval. This is what --resume loads. Big: ~380 MB at 31M params,
            because AdamW keeps two moment buffers per parameter.
  best.pt   model weights only, written when validation loss improves. ~130 MB.
            This is what you actually ship.
  step_N.pt model weights only, rotated so only the newest few are kept. Gives
            you a window of recent history without unbounded disk growth.

Why atomic writes
-----------------
A checkpoint written directly to its final path is corrupt if the process dies
mid-write, and an interrupted training run is exactly when you need the
checkpoint most. Writing to a sibling temp file and then os.replace() makes the
swap atomic on the same filesystem, so last.pt is always either the previous
complete state or the new complete state -- never half of one.

Worth knowing: the same applies to Ctrl+C. train_pretrain installs a SIGINT
handler that finishes the current step, writes a checkpoint, and exits, so an
accidental interrupt costs you one step rather than the whole run.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

CKPT_VERSION = 1


# --------------------------------------------------------------------------
#  Atomic write
# --------------------------------------------------------------------------


def atomic_save(obj: Any, path: str | Path) -> Path:
    """torch.save to a temp sibling, then atomically rename into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
#  Save
# --------------------------------------------------------------------------


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        # A mismatched RNG state is not worth failing a resume over.
        pass


def save_full(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
) -> Path:
    """Everything needed to continue training exactly where it stopped."""
    payload: Dict[str, Any] = {
        "version": CKPT_VERSION,
        "kind": "full",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "model_config": model_config,
        "train_config": train_config,
        "rng": _rng_state(),
    }
    if extras:
        payload.update(extras)
    return atomic_save(payload, path)


def save_model_only(
    path: str | Path,
    *,
    model: nn.Module,
    step: int,
    best_val: float,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
) -> Path:
    """Weights plus the metadata needed to rebuild the model. No optimiser state."""
    payload: Dict[str, Any] = {
        "version": CKPT_VERSION,
        "kind": "model",
        "model": model.state_dict(),
        "step": step,
        "best_val": best_val,
        "model_config": model_config,
        "train_config": train_config,
    }
    if extras:
        payload.update(extras)
    return atomic_save(payload, path)


# --------------------------------------------------------------------------
#  Load
# --------------------------------------------------------------------------


def peek(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Read a checkpoint's metadata without loading the weights."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return {k: v for k, v in ckpt.items() if k not in ("model", "optimizer")}


def load_into(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    map_location: str = "cpu",
    load_optimizer: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """
    Restore weights (and optimiser state when present) into live objects.

    Returns the checkpoint dict; callers read `step` and `best_val` from it.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match the model being built.\n"
            f"  missing keys   : {list(missing)[:5]}\n"
            f"  unexpected keys: {list(unexpected)[:5]}\n"
            "This almost always means the model config changed since the save."
        )
    if optimizer is not None and load_optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if restore_rng:
        _restore_rng(ckpt.get("rng"))
    return ckpt


# --------------------------------------------------------------------------
#  Rotation
# --------------------------------------------------------------------------


def rotate_checkpoints(
    out_dir: str | Path, prefix: str = "step_", keep: int = 3
) -> List[Path]:
    """
    Keep only the newest `keep` step_*.pt files.

    Sorted by step number parsed from the filename rather than mtime, so copying
    the directory around cannot reorder history.
    """
    out_dir = Path(out_dir)
    files = list(out_dir.glob(f"{prefix}*.pt"))

    def step_of(p: Path) -> int:
        stem = p.stem[len(prefix):]
        return int(stem) if stem.isdigit() else -1

    files.sort(key=step_of)
    removed: List[Path] = []
    for f in files[:-keep] if keep > 0 else files:
        f.unlink(missing_ok=True)
        removed.append(f)
    return removed


def find_latest(out_dir: str | Path, prefix: str = "step_") -> Optional[Path]:
    out_dir = Path(out_dir)
    files = list(out_dir.glob(f"{prefix}*.pt"))

    def step_of(p: Path) -> int:
        stem = p.stem[len(prefix):]
        return int(stem) if stem.isdigit() else -1

    if not files:
        return None
    return max(files, key=step_of)


def resolve_resume(out_dir: str | Path, spec: str) -> Optional[Path]:
    """
    Interpret a --resume value.

      "auto"   -> last.pt if it exists, else the newest step_*.pt, else None
      "none"   -> None (start fresh)
      <path>   -> that path, which must exist
    """
    out_dir = Path(out_dir)
    if spec in ("none", "", "off"):
        return None
    if spec != "auto":
        p = Path(spec)
        if not p.exists():
            raise FileNotFoundError(f"--resume {spec!r} does not exist")
        return p

    last = out_dir / "last.pt"
    if last.exists():
        return last
    return find_latest(out_dir)


def describe(path: str | Path) -> str:
    meta = peek(path)
    bits = [
        f"{Path(path).name}",
        f"kind={meta.get('kind', '?')}",
        f"step={meta.get('step', '?')}",
    ]
    if "best_val" in meta:
        bv = meta["best_val"]
        bits.append(f"best_val={'n/a' if bv is None else f'{bv:.4f}'}")
    mc = meta.get("model_config") or {}
    if mc:
        bits.append(
            f"vocab={mc.get('vocab_size')} d={mc.get('d_model')} "
            f"L={mc.get('n_layer')} ctx={mc.get('ctx_len')}"
        )
    return "  ".join(bits)
