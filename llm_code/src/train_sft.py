#!/usr/bin/env python
"""
SFT (supervised fine-tuning): turn a story-continuing base model into one that
answers.

What changes versus pretraining
-------------------------------
Only three things, and everything else is reused:

  1. The data is conversations, not a flat token stream.
  2. The loss is MASKED: only assistant turns are supervised. User turns and
     system turns are context, not targets. On a 6-30M parameter model this
     matters a lot -- predicting the user's own words would consume most of the
     capacity for no benefit.
  3. The learning rate is much lower. The model already speaks English; SFT is
     steering behaviour, not teaching language. A big LR here destroys the
     pretrained representations and the model gets worse, not better.

Everything else -- the model, the optimiser, checkpointing, atomic saves,
SIGINT handling, KV-cache generation -- is the same code as pretraining.

Reading the results
-------------------
The checkpoint shown at step 0 (before any training) is the pretrained base
model. Ask it a question and it CONTINUES the question as if it were story text.
After a few hundred steps it should start producing replies instead. That
behaviour change -- not the loss number -- is the milestone.

Usage
-----
    # fine-tune the fully-trained tiny base model
    python -m src.train_sft --init runs/tiny_full/best.pt \
        --tokenizer data/tokenizer/tiny_tokenizer.json \
        --data-glob "data/sft/shard_*.jsonl" --out-dir runs/tiny_sft
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import (  # noqa: E402
    describe,
    resolve_resume,
    rotate_checkpoints,
    save_full,
    save_model_only,
)
from src.config import ModelConfig, TrainConfig  # noqa: E402
from src.model import GPT, compute_loss  # noqa: E402
from src.paths import describe_missing, resolve  # noqa: E402
from src.sft_data import SFTDataset, dedupe_by_question, load_jsonl, split_samples  # noqa: E402
from src.template import build_default_system_variants, build_prompt  # noqa: E402

_INTERRUPT = False


def _install_sigint() -> None:
    def handler(signum, frame):  # noqa: ARG001
        global _INTERRUPT
        if _INTERRUPT:
            print("\n  second interrupt -- exiting now", flush=True)
            raise SystemExit(130)
        _INTERRUPT = True
        print(
            "\n  interrupt received: finishing this step, saving a checkpoint, "
            "then exiting. Ctrl+C again to abort immediately.",
            flush=True,
        )

    signal.signal(signal.SIGINT, handler)


def get_lr(step: int, warmup: int, max_iters: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_iters:
        return min_lr
    ratio = (step - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


# --------------------------------------------------------------------------
#  Evaluation and sampling
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: GPT, loader, device, amp) -> tuple[float, float]:
    """
    Returns (masked loss, accuracy on supervised tokens).

    Accuracy is worth watching: loss can fall while the model merely gets the
    easy tokens right. Per-token accuracy on supervised positions is the
    blunter, more honest number.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    counted = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        with amp:
            logits, _ = model(x)
        loss = compute_loss(logits, y, m)
        total_loss += loss.item()
        n_batches += 1

        pred = logits.argmax(dim=-1)
        sel = m > 0
        correct += (pred[sel] == y[sel]).sum().item()
        counted += int(sel.sum().item())
    model.train()
    return total_loss / max(n_batches, 1), correct / max(counted, 1)


@torch.no_grad()
def ask(
    model: GPT,
    tok,
    question: str,
    device,
    *,
    system: str | None = None,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
) -> str:
    """Run one chat turn through the shared template. This is the payoff."""
    from src.template import EOS_ID

    prompt = build_prompt(
        [{"role": "user", "content": question}],
        system=system,
        add_generation_prompt=True,
    )
    ids = tok.encode(prompt)
    if len(ids) >= model.cfg.ctx_len:
        ids = ids[-(model.cfg.ctx_len - 1) :]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x, max_new_tokens, temperature=temperature, top_p=0.9,
        repetition_penalty=1.1, eos_id=EOS_ID,
    )
    reply = out[0].tolist()[len(ids) :]
    return tok.decode(reply, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--init", default="runs/tiny_full/best.pt",
                    help="pretrained checkpoint to start from")
    ap.add_argument("--tokenizer", default="data/tokenizer/tiny_tokenizer.json")
    ap.add_argument("--data-glob", default="data/sft/shard_*.jsonl")
    ap.add_argument("--out-dir", default="runs/tiny_sft")
    ap.add_argument("--resume", default="auto", help="auto | none | path")

    ap.add_argument("--epochs", type=float, default=10.0,
                    help="10 was the best val point on the 500-dialogue set; "
                         "more epochs overfit (train loss keeps falling, val rises)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min-lr", type=float, default=1.5e-5)
    ap.add_argument("--warmup-iters", type=int, default=30)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="SFT sets are small; a little dropout fights overfitting")
    ap.add_argument("--val-fraction", type=float, default=0.06)
    ap.add_argument("--max-len", type=int, default=0, help="0 = use the model's ctx_len")
    ap.add_argument("--max-iters", type=int, default=0, help="0 = derive from --epochs")
    ap.add_argument("--no-dedupe", action="store_true")

    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--sample-interval", type=int, default=100)
    ap.add_argument("--keep-last-n-ckpts", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _install_sigint()
    torch.manual_seed(args.seed)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer ------------------------------------------------------
    from src.tokenizer import BPETokenizer

    tk_path = resolve(args.tokenizer)
    if not tk_path.exists():
        raise SystemExit(f"tokenizer not found: {describe_missing(tk_path)}")
    tok = BPETokenizer.load(tk_path)

    # ---- device / amp ---------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = args.dtype
    if dtype == "bf16" and (device.type == "cpu" or not torch.cuda.is_bf16_supported()):
        dtype = "fp32"
    amp = torch.autocast(
        device_type=device.type,
        dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype],
        enabled=(dtype != "fp32"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "fp16" and device.type == "cuda"))

    # ---- data -----------------------------------------------------------
    import glob as _glob

    files = sorted(_glob.glob(str(resolve(args.data_glob))))
    if not files:
        raise SystemExit(
            f"no SFT data matched {args.data_glob!r}\n"
            f"    looked under {resolve('.')}\n"
            f"    generate some first, or check --data-glob"
        )
    samples = load_jsonl(files)
    n_raw = len(samples)
    if not args.no_dedupe:
        samples = dedupe_by_question(samples)
    train_s, val_s = split_samples(samples, args.val_fraction, args.seed)

    print("=" * 74)
    print(f"  SFT data    : {len(files)} files, {n_raw} dialogues"
          + (f" -> {len(samples)} after de-duplication" if len(samples) != n_raw else ""))
    print(f"  split       : train {len(train_s)} / val {len(val_s)}")
    print(f"  tokenizer   : {tk_path.name}  (vocab {tok.vocab_size:,})")
    # ---- measure the real conversation length FIRST -----------------------
    # Building the report dataset at a pessimistic padding guess (say 2048)
    # makes "supervised share" look absurdly low, because the denominator is
    # mostly padding. Measure the true length, then pad to exactly that.
    if args.max_len:
        real_len = args.max_len
    else:
        from src.sft_data import build_example

        lens = [
            len(build_example(tok, s["messages"], s.get("system"))[0])
            for s in (train_s + val_s)
        ]
        real_len = max(64, max(lens))
    print(f"  max_len     : {real_len} (padded to the longest conversation)")

    train_ds = SFTDataset(train_s, tok, real_len, verbose=True)
    val_ds = SFTDataset(val_s, tok, real_len, verbose=False)
    print(f"  sequences   : train {len(train_ds)} / val {len(val_ds)}")

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
        generator=gen, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = max(1, len(train_loader))
    max_iters = args.max_iters or max(1, int(steps_per_epoch * args.epochs))

    # ---- model from the pretrained checkpoint ---------------------------
    init_path = resolve(args.init)
    if not init_path.exists():
        raise SystemExit(
            f"pretrained checkpoint not found: {describe_missing(init_path)}\n"
            "    SFT starts from a base model; train one first:\n"
            "      run.cmd -m src.train_pretrain --preset tiny --max-iters 7700 --out-dir runs/tiny_full"
        )
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt:
        raise SystemExit(f"{init_path} has no 'model_config'; cannot rebuild the architecture")
    cfg = ModelConfig(**ckpt["model_config"])
    if cfg.vocab_size != tok.vocab_size:
        raise SystemExit(
            f"tokenizer/model mismatch: tokenizer vocab is {tok.vocab_size}, "
            f"model expects {cfg.vocab_size}"
        )
    cfg = ModelConfig(**{**asdict(cfg), "dropout": args.dropout})

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()

    tcfg = TrainConfig(
        lr=args.lr, min_lr=args.min_lr, warmup_iters=args.warmup_iters,
        max_iters=max_iters, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, grad_accum=args.grad_accum,
        micro_batch=args.batch_size, seq_len=train_ds.max_len,
        out_dir=str(out_dir), eval_interval=args.eval_interval,
        save_interval=args.save_interval, keep_last_n_ckpts=args.keep_last_n_ckpts,
        log_interval=args.log_interval, sample_interval=args.sample_interval,
        seed=args.seed,
    )
    optimizer = model.configure_optimizers(tcfg)

    start_step = 0
    best_val = float("inf")
    resume_path = resolve_resume(out_dir, args.resume)
    if resume_path is not None:
        from src.checkpoint import load_into

        print(f"  resuming    : {resume_path}")
        rc = load_into(resume_path, model, optimizer, map_location=device)
        start_step = int(rc.get("step", 0))
        best_val = float(rc.get("best_val", float("inf")) or float("inf"))

    n_params = model.num_params()
    print(f"  base model  : {describe(init_path)}")
    print(f"  params      : {n_params/1e6:.3f}M   dropout={args.dropout}")
    print(f"  device      : {device}  dtype={dtype}")
    print(f"  steps       : {start_step} -> {max_iters}  ({args.epochs} epochs x {steps_per_epoch}/epoch)")
    print(f"  batch       : {args.batch_size} x grad_accum {args.grad_accum}")
    print(f"  lr          : {args.lr} -> {args.min_lr} (warmup {args.warmup_iters})")
    print(f"  out dir     : {out_dir}")
    print("=" * 74, flush=True)

    probe_questions = [
        "What is an adversarial example?",
        "How do I evaluate robustness?",
        "What is the difference between white-box and black-box attacks?",
    ]
    systems = build_default_system_variants()

    def show(model_: GPT, tag: str) -> None:
        print(f"\n  --- sample @ {tag} ---", flush=True)
        for q in probe_questions:
            try:
                a = ask(model_, tok, q, device, system=systems[0], max_new_tokens=40)
            except Exception as e:
                a = f"(failed: {e})"
            print(f"    Q: {q}")
            print(f"    A: {a}", flush=True)
        print()

    # ---- training loop --------------------------------------------------
    log_path = out_dir / "train_log.csv"
    mode = "a" if (start_step > 0 and log_path.exists()) else "w"
    log_fh = log_path.open(mode, newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    if mode == "w":
        writer.writerow(["step", "epoch", "lr", "train_loss", "val_loss", "val_acc", "elapsed_s"])

    print("\n  step | epoch |   lr     | train    | val      | acc    | note")
    print("  -----+-------+----------+----------+----------+--------+------")

    train_iter = iter(train_loader)
    t0 = time.time()
    running, running_n = 0.0, 0
    step = start_step

    try:
        while step < max_iters:
            lr = get_lr(step, args.warmup_iters, max_iters, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum = 0.0
            for _ in range(max(1, args.grad_accum)):
                try:
                    x, y, m = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y, m = next(train_iter)
                x, y, m = x.to(device), y.to(device), m.to(device)
                with amp:
                    logits, _ = model(x)
                    loss = compute_loss(logits, y, m) / max(1, args.grad_accum)
                scaler.scale(loss).backward()
                accum += loss.item()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += accum
            running_n += 1
            step += 1
            epoch = step / steps_per_epoch

            if step % args.log_interval == 0 or step == start_step + 1:
                avg = running / max(running_n, 1)
                print(
                    f"  {step:>5} | {epoch:>5.1f} | {lr:.2e} | {avg:8.4f} |          |        |",
                    flush=True,
                )
                writer.writerow([step, f"{epoch:.2f}", f"{lr:.6e}", f"{avg:.6f}", "", "",
                                 f"{time.time()-t0:.2f}"])
                log_fh.flush()
                running, running_n = 0.0, 0

            if step % args.eval_interval == 0 or step == max_iters:
                vl, va = evaluate(model, val_loader, device, amp)
                flag = ""
                if vl < best_val:
                    best_val = vl
                    save_model_only(
                        out_dir / "best.pt", model=model, step=step, best_val=best_val,
                        model_config=asdict(cfg), train_config=asdict(tcfg),
                    )
                    flag = "  <- best, saved best.pt"
                print(f"        val {vl:.4f}  acc {va*100:.1f}%   best {best_val:.4f}{flag}", flush=True)
                writer.writerow([step, f"{epoch:.2f}", f"{lr:.6e}", "", f"{vl:.6f}",
                                 f"{va:.4f}", f"{time.time()-t0:.2f}"])
                log_fh.flush()

            if step % args.sample_interval == 0:
                show(model, f"step {step}")

            if step % args.save_interval == 0:
                save_full(
                    out_dir / "last.pt", model=model, optimizer=optimizer, step=step,
                    best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg),
                )
                save_model_only(
                    out_dir / f"step_{step:07d}.pt", model=model, step=step,
                    best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg),
                )
                rotate_checkpoints(out_dir, "step_", args.keep_last_n_ckpts)

            if _INTERRUPT:
                print("\n  saving checkpoint before exit...", flush=True)
                break
    except Exception as e:
        print(f"\n  SFT ERROR at step {step}: {type(e).__name__}: {e}", flush=True)
        save_full(
            out_dir / "last.pt", model=model, optimizer=optimizer, step=step,
            best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg),
        )
        raise
    finally:
        log_fh.close()

    save_full(out_dir / "last.pt", model=model, optimizer=optimizer, step=step,
              best_val=best_val, model_config=asdict(cfg), train_config=asdict(tcfg))
    save_model_only(out_dir / "final.pt", model=model, step=step, best_val=best_val,
                    model_config=asdict(cfg), train_config=asdict(tcfg))

    print("=" * 74)
    print(f"  finished at step {step}  ({step/steps_per_epoch:.1f} epochs)")
    print(f"  elapsed      : {time.time()-t0:.1f}s")
    print(f"  best val     : {best_val:.4f}")
    print(f"  best.pt      : {out_dir/'best.pt' if best_val < float('inf') else '(none)'}")
    print(f"  log          : {log_path}")
    print("=" * 74)

    show(model, f"final step {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
