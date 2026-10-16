#!/usr/bin/env python
"""
Pretraining loop: next-token prediction on a memory-mapped token stream.

Everything here is standard except the parts that are easy to get wrong:

  * Effective batch size is micro_batch * grad_accum * seq_len. Micro-batching
    keeps activation memory inside the 8 GB budget while the accumulated
    gradient still corresponds to the large batch that stable training needs.
  * The LR schedule is linear warmup then cosine decay to min_lr. Warmup exists
    because Adam's second-moment estimate is unreliable for the first few
    hundred steps; a full-size LR on top of that estimate reliably diverges.
  * bf16 needs no GradScaler (its exponent range matches fp32). fp16 does, so
    the scaler is created for fp16 only.
  * Checkpoints are written atomically and SIGINT is handled, so Ctrl+C costs
    one step rather than the whole run. See src/checkpoint.py.
  * Loss is reported both as the raw value and as bits-per-token-ish intuition:
    at initialisation a correctly configured model scores ln(vocab_size).

Usage
-----
    # smoke test: tiny config, a couple of hundred steps
    python -m src.train_pretrain --preset tiny --max-iters 200 \
        --train-bin data/processed/tiny_train.bin --out-dir runs/tiny

    # real run
    python -m src.train_pretrain --preset base --resume auto
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import (  # noqa: E402
    describe,
    resolve_resume,
    rotate_checkpoints,
    save_full,
    save_model_only,
)
from src.config import BASE, PRESETS, TINY, ModelConfig, TrainConfig, get_preset  # noqa: E402
from src.dataset import TokenSlice, TokenizedDataset, dtype_for_vocab  # noqa: E402
from src.model import GPT  # noqa: E402
from src.paths import ROOT, describe_missing, resolve  # noqa: E402

# --------------------------------------------------------------------------
#  Interrupt handling
# --------------------------------------------------------------------------

_INTERRUPT = False


def _install_sigint() -> None:
    def handler(signum, frame):  # noqa: ARG001
        global _INTERRUPT
        if _INTERRUPT:
            print("\n  second interrupt -- exiting immediately", flush=True)
            raise SystemExit(130)
        _INTERRUPT = True
        print(
            "\n  interrupt received: finishing the current step, saving a "
            "checkpoint, then exiting. Press Ctrl+C again to abort now.",
            flush=True,
        )

    signal.signal(signal.SIGINT, handler)


# --------------------------------------------------------------------------
#  LR schedule
# --------------------------------------------------------------------------


def get_lr(step: int, t: TrainConfig) -> float:
    if step < t.warmup_iters:
        return t.lr * (step + 1) / max(1, t.warmup_iters)
    if step >= t.max_iters:
        return t.min_lr
    ratio = (step - t.warmup_iters) / max(1, t.max_iters - t.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return t.min_lr + coeff * (t.lr - t.min_lr)


# --------------------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(model: GPT, loader, n_batches: int, device, amp) -> float:
    model.eval()
    it = iter(loader)
    total = 0.0
    seen = 0
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp:
            _, loss = model(x, targets=y)
        total += loss.item()
        seen += 1
    model.train()
    return total / max(seen, 1)


@torch.no_grad()
def sample_text(model: GPT, prompt_ids, device, max_new_tokens: int = 60) -> str:
    """Generate a short sample. Decoding happens in the caller (tokenizer lives there)."""
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens, temperature=0.8, top_p=0.95, repetition_penalty=1.15
    )
    model.train()
    return out[0].tolist()


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    ap.add_argument("--train-bin", default=None, help="default: <data-dir>/train.bin")
    ap.add_argument("--val-bin", default=None, help="default: <data-dir>/val.bin")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--val-fraction", type=float, default=0.005,
                    help="used only when no val.bin exists: hold out this slice of train.bin")
    ap.add_argument("--out-dir", default=None, help="default: runs/<preset>")
    ap.add_argument("--resume", default="auto", help="auto | none | path/to/ckpt.pt")
    ap.add_argument("--init", default=None,
                    help="start from pretrained WEIGHTS but reset step counter and "
                         "optimiser. This is what domain-adaptive pretraining needs: "
                         "continue a pretrained model on a new corpus without "
                         "inheriting the old LR schedule.")

    # overrides, so the same script covers smoke tests and real runs
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=None)
    ap.add_argument("--save-interval", type=int, default=None)
    ap.add_argument("--sample-interval", type=int, default=None)
    ap.add_argument("--warmup-iters", type=int, default=None)
    ap.add_argument("--keep-last-n-ckpts", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sample-prompt", default="Once upon a time")
    ap.add_argument("--no-sample", action="store_true",
                    help="disable periodic text samples (they cost a little time)")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json",
                    help="used for the periodic sample decode; missing file just disables sampling")
    ap.add_argument("--max-steps-time", type=float, default=0.0,
                    help="stop after this many seconds (0 = no limit). Useful for smoke runs.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _install_sigint()

    cfg: ModelConfig = get_preset(args.preset)
    tcfg = TrainConfig()

    # ---- apply overrides ------------------------------------------------
    if args.max_iters is not None:
        tcfg.max_iters = args.max_iters
    if args.micro_batch is not None:
        tcfg.micro_batch = args.micro_batch
    if args.grad_accum is not None:
        tcfg.grad_accum = args.grad_accum
    if args.lr is not None:
        tcfg.lr = args.lr
        tcfg.min_lr = args.lr / 10
    if args.seq_len is not None:
        tcfg.seq_len = args.seq_len
    if args.log_interval is not None:
        tcfg.log_interval = args.log_interval
    if args.eval_interval is not None:
        tcfg.eval_interval = args.eval_interval
    if args.eval_iters is not None:
        tcfg.eval_iters = args.eval_iters
    if args.save_interval is not None:
        tcfg.save_interval = args.save_interval
    if args.sample_interval is not None:
        tcfg.sample_interval = args.sample_interval
    if args.warmup_iters is not None:
        tcfg.warmup_iters = args.warmup_iters
    if args.keep_last_n_ckpts is not None:
        tcfg.keep_last_n_ckpts = args.keep_last_n_ckpts
    if args.dtype is not None:
        tcfg.dtype = args.dtype
    if args.grad_checkpointing:
        tcfg.grad_checkpointing = True
    if args.compile:
        tcfg.compile = True
    tcfg.seed = args.seed
    tcfg.sample_prompt = args.sample_prompt

    # The model's context window and the training sequence length must agree;
    # otherwise RoPE is built for one length and asked for another.
    if tcfg.seq_len > cfg.ctx_len:
        print(f"  reducing seq_len {tcfg.seq_len} -> ctx_len {cfg.ctx_len}")
        tcfg.seq_len = cfg.ctx_len
    elif tcfg.seq_len < cfg.ctx_len:
        cfg = replace(cfg, ctx_len=tcfg.seq_len)

    out_dir = resolve(args.out_dir or f"runs/{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg.out_dir = str(out_dir)

    # ---- device ---------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if tcfg.dtype == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("  bf16 unsupported on this GPU; falling back to fp16")
            tcfg.dtype = "fp16"
    if device.type == "cpu" and tcfg.dtype != "fp32":
        print("  CPU detected; forcing fp32")
        tcfg.dtype = "fp32"

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[tcfg.dtype]
    amp = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=tcfg.dtype != "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg.dtype == "fp16" and device.type == "cuda"))

    # ---- data -----------------------------------------------------------
    # Paths are anchored to the project root, so PyCharm run configurations do
    # not need a correctly-set working directory.
    data_dir = resolve(args.data_dir)
    train_bin = resolve(args.train_bin) if args.train_bin else data_dir / "train.bin"
    val_bin = resolve(args.val_bin) if args.val_bin else data_dir / "val.bin"

    if not train_bin.exists():
        raise SystemExit(
            f"training binaries not found: {describe_missing(train_bin)}\n"
            "    build them first:  run.cmd scripts/prepare_pretrain_data.py --help"
        )

    dtype = dtype_for_vocab(cfg.vocab_size)
    full_train = TokenizedDataset(train_bin, tcfg.seq_len, dtype)

    if val_bin.exists():
        train_ds: TokenizedDataset = full_train
        val_ds: TokenizedDataset = TokenizedDataset(val_bin, tcfg.seq_len, dtype)
        val_source = f"{val_bin.name} ({val_ds.n_tokens:,} tokens)"
    else:
        # No validation file: hold out a slice of the training tokens. Slicing at
        # TOKEN granularity (from the end) keeps train and val disjoint without
        # copying anything.
        n_val = max(tcfg.seq_len * 64, int(full_train.n_tokens * args.val_fraction))
        n_val = min(n_val, full_train.n_tokens // 10)
        split = full_train.n_tokens - n_val
        train_ds = TokenSlice(full_train, 0, split)
        val_ds = TokenSlice(full_train, split, n_val)
        val_source = f"held-out tail slice of train.bin ({n_val:,} tokens)"

    from torch.utils.data import DataLoader

    gen = torch.Generator()
    gen.manual_seed(tcfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg.micro_batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        generator=gen, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.micro_batch, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ---- model ----------------------------------------------------------
    torch.manual_seed(tcfg.seed)
    model = GPT(cfg).to(device)
    model.grad_checkpointing = tcfg.grad_checkpointing
    optimizer = model.configure_optimizers(tcfg)

    start_step = 0
    best_val = float("inf")

    if args.init:
        init_path = resolve(args.init)
        if not init_path.exists():
            raise SystemExit(f"--init checkpoint not found: {describe_missing(init_path)}")
        _ck = torch.load(init_path, map_location="cpu", weights_only=False)
        if "model_config" not in _ck:
            raise SystemExit(f"{init_path} has no 'model_config'")
        _icfg = ModelConfig(**_ck["model_config"])
        if _icfg.vocab_size != cfg.vocab_size:
            raise SystemExit(
                f"--init/model mismatch: checkpoint vocab is {_icfg.vocab_size} "
                f"but the preset expects {cfg.vocab_size}"
            )
        model.load_state_dict(_ck["model"])
        print(f"initialised weights from {init_path}  (step reset to 0, fresh optimiser)")
        del _ck

    # --init wins over --resume: the two mean different things.
    resume_path = None if args.init else resolve_resume(out_dir, args.resume)
    if resume_path is not None:
        from src.checkpoint import load_into

        print(f"resuming from {resume_path}")
        ckpt = load_into(resume_path, model, optimizer, map_location=device)
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")) or float("inf"))
        print(f"  {describe(resume_path)}")

    if tcfg.compile and hasattr(torch, "compile"):
        print("  torch.compile enabled (first step will be slow)")
        model = torch.compile(model)  # type: ignore[assignment]

    # ---- banner ---------------------------------------------------------
    n_params = model.num_params()
    print("=" * 72)
    print(f"  preset          : {args.preset}")
    print(f"  device          : {device}  dtype={tcfg.dtype}")
    if device.type == "cuda":
        print(f"  gpu             : {torch.cuda.get_device_name(0)}")
    print(f"  model           : {cfg.summary()}")
    print(f"  params          : {n_params/1e6:.3f}M")
    print(f"  train tokens    : {train_ds.n_tokens:,}")
    print(f"  val tokens      : {val_ds.n_tokens:,}  ({val_source})")
    print(f"  seq_len         : {tcfg.seq_len}   micro_batch={tcfg.micro_batch}  grad_accum={tcfg.grad_accum}")
    print(f"  tokens per step : {tcfg.tokens_per_step:,}")
    print(f"  steps           : {start_step} -> {tcfg.max_iters}  "
          f"({(tcfg.max_iters - start_step) * tcfg.tokens_per_step:,} more tokens)")
    print(f"  lr              : {tcfg.lr} -> {tcfg.min_lr} (warmup {tcfg.warmup_iters})")
    print(f"  out dir         : {out_dir}")
    print("=" * 72, flush=True)

    log_path = out_dir / "train_log.csv"
    new_log = not log_path.exists() or start_step == 0
    log_fh = log_path.open("a" if not new_log else "w", newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    if new_log:
        writer.writerow(["step", "lr", "train_loss", "val_loss", "tokens", "elapsed_s", "tok_per_s"])

    # Generation samples go to their own file as well as stdout.
    # Without this, only whoever owns the process stdout can see them: a CSV
    # tail shows the loss but never shows what the model actually writes, which
    # is the single most useful thing to watch during a long run.
    samples_path = out_dir / "samples.log"
    samples_fh = samples_path.open("a" if start_step > 0 else "w", encoding="utf-8")
    samples_fh.write(f"# generation samples for {out_dir.name}\n")
    samples_fh.flush()

    # prompt ids for periodic sampling
    prompt_ids = None
    sampling_tok = None
    try:
        from src.tokenizer import BPETokenizer

        tk_path = resolve(args.tokenizer)
        if tk_path.exists():
            sampling_tok = BPETokenizer.load(tk_path)
            prompt_ids = sampling_tok.encode(args.sample_prompt)
            print(f"  sample prompt   : {args.sample_prompt!r} -> {prompt_ids}")
        else:
            print(f"  (no tokenizer at {tk_path}; periodic sampling disabled)")
    except Exception as e:  # sampling is a nicety, never fail the run over it
        print(f"  (sampling disabled: {e})")

    # ---- training loop --------------------------------------------------
    model.train()
    train_iter = iter(train_loader)
    t_start = time.time()
    tokens_done = start_step * tcfg.tokens_per_step
    running_loss = 0.0
    running_n = 0

    print("\n  step |   lr     | train    | val      | tok/s   | elapsed | note")
    print("  -----+----------+----------+----------+---------+---------+------")

    step = start_step
    try:
        while step < tcfg.max_iters:
            lr = get_lr(step, tcfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- gradient accumulation ---------------------------------
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(tcfg.grad_accum):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with amp:
                    _, loss = model(x, targets=y)
                    loss = loss / tcfg.grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            if tcfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += accum_loss
            running_n += 1
            step += 1
            tokens_done += tcfg.tokens_per_step

            # ---- logging ------------------------------------------------
            if step % tcfg.log_interval == 0 or step == start_step + 1:
                elapsed = time.time() - t_start
                # tokens_done is CUMULATIVE from step 0, but elapsed is measured
                # from the resume point. Dividing one by the other after a resume
                # reports a throughput inflated by (total / this-run) -- about 3x
                # on a run resumed a third of the way in. Subtract the tokens that
                # were already done before this process started.
                tokens_before = start_step * tcfg.tokens_per_step
                tps = (tokens_done - tokens_before) / max(elapsed, 1e-6)
                avg = running_loss / max(running_n, 1)
                note = ""
                if step % tcfg.eval_interval == 0 and step > start_step:
                    note = "eval"
                print(
                    f"  {step:>5} | {lr:.2e} | {avg:8.4f} |          | "
                    f"{tps:7.0f} | {elapsed:7.1f} | {note}",
                    flush=True,
                )
                writer.writerow([step, f"{lr:.6e}", f"{avg:.6f}", "", tokens_done,
                                 f"{elapsed:.2f}", f"{tps:.1f}"])
                log_fh.flush()
                running_loss = 0.0
                running_n = 0

            # ---- evaluation ---------------------------------------------
            val_loss = None
            if step % tcfg.eval_interval == 0 and step > start_step:
                val_loss = estimate_loss(model, val_loader, tcfg.eval_iters, device, amp)
                flag = ""
                if val_loss < best_val:
                    best_val = val_loss
                    save_model_only(
                        out_dir / "best.pt", model=model, step=step, best_val=best_val,
                        model_config=asdict(cfg), train_config=asdict(tcfg),
                    )
                    flag = "  <- new best, saved best.pt"
                print(f"        val loss {val_loss:.4f}   best {best_val:.4f}{flag}", flush=True)
                writer.writerow([step, f"{lr:.6e}", "", f"{val_loss:.6f}", tokens_done,
                                 f"{time.time()-t_start:.2f}", ""])
                log_fh.flush()

            # ---- sampling -----------------------------------------------
            if (
                prompt_ids
                and not args.no_sample
                and step % tcfg.sample_interval == 0
                and step > start_step
            ):
                try:
                    out_ids = sample_text(model, prompt_ids, device)
                    text = (
                        sampling_tok.decode(out_ids)
                        if sampling_tok is not None
                        else str(out_ids[:40])
                    )
                    print(f"        sample: {text[:200]!r}", flush=True)
                    val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
                    samples_fh.write(
                        f"=== step {step}  val={val_str}\n{text}\n\n"
                    )
                    samples_fh.flush()
                except Exception as e:
                    print(f"        (sample failed: {e})")

            # ---- checkpoints --------------------------------------------
            if step % tcfg.save_interval == 0 and step > start_step:
                save_full(
                    out_dir / "last.pt", model=model, optimizer=optimizer, step=step,
                    best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg),
                )
                save_model_only(
                    out_dir / f"step_{step:07d}.pt", model=model, step=step,
                    best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg),
                )
                removed = rotate_checkpoints(out_dir, "step_", tcfg.keep_last_n_ckpts)
                print(
                    f"        checkpoint saved at step {step}"
                    + (f" (rotated out {len(removed)})" if removed else ""),
                    flush=True,
                )

            # ---- stop conditions ----------------------------------------
            if _INTERRUPT:
                print("\n  saving checkpoint before exiting...", flush=True)
                break
            if args.max_steps_time and (time.time() - t_start) >= args.max_steps_time:
                print(f"\n  time limit {args.max_steps_time}s reached", flush=True)
                break

    except SystemExit:
        raise
    except Exception as e:
        print(f"\n  TRAINING ERROR at step {step}: {type(e).__name__}: {e}", flush=True)
        print("  saving a checkpoint so the run can be resumed...", flush=True)
        save_full(
            out_dir / "last.pt", model=model, optimizer=optimizer, step=step,
            best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg),
        )
        raise
    finally:
        log_fh.close()

    # ---- final save -----------------------------------------------------
    final_path = save_full(
        out_dir / "last.pt", model=model, optimizer=optimizer, step=step,
        best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg),
    )
    final_model = save_model_only(
        out_dir / "final.pt", model=model, step=step, best_val=best_val,
        model_config=asdict(cfg), train_config=asdict(tcfg),
    )
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"  finished at step {step}")
    print(f"  tokens this run : {tokens_done - start_step * tcfg.tokens_per_step:,}")
    print(f"  elapsed         : {elapsed:.1f}s")
    print(f"  best val loss   : {best_val:.4f}")
    print(f"  last.pt         : {final_path}")
    print(f"  final.pt        : {final_model}")
    if best_val < float("inf"):
        print(f"  best.pt         : {out_dir / 'best.pt'}")
    print(f"  log             : {log_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
