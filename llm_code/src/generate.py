#!/usr/bin/env python
"""
Sample text from a trained checkpoint.

This is the fastest way to answer "is it actually learning?". A loss curve tells
you the model is improving; only generated text tells you WHAT it learned. Watch
for the progression as training runs: random punctuation -> real words -> local
grammar -> globally coherent sentences.

The model here is a base (pretrained) model, not a chat model, so it CONTINUES
a prompt rather than answering it. Asking it a question will produce more text
that looks like the training corpus, not an answer. That is expected until SFT.

Usage
-----
    # single prompt
    python -m src.generate --ckpt runs/tiny_smoke/best.pt \
        --tokenizer data/tokenizer/tiny_tokenizer.json \
        --prompt "Once upon a time"

    # compare sampling settings
    python -m src.generate --prompt "Once upon a time" --temperature 0.0
    python -m src.generate --prompt "Once upon a time" --temperature 1.0 --top-p 0.9

    # explore interactively
    python -m src.generate --interactive
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ModelConfig  # noqa: E402
from src.model import GPT  # noqa: E402
from src.paths import describe_missing, resolve  # noqa: E402
from src.template import DEFAULT_SYSTEM, EOS_ID, build_prompt  # noqa: E402


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[GPT, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt:
        raise SystemExit(
            f"{ckpt_path} has no 'model_config'; cannot rebuild the architecture.\n"
            "Was it written by src/train_pretrain.py?"
        )
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint does not match the architecture:\n"
            f"  missing   : {list(missing)[:5]}\n"
            f"  unexpected: {list(unexpected)[:5]}"
        )
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "rng")}
    return model, meta


def describe_ckpt(path: Path, meta: dict) -> str:
    bits = [f"{path.name}"]
    if "kind" in meta:
        bits.append(f"kind={meta['kind']}")
    if "step" in meta:
        bits.append(f"step={meta['step']}")
    bv = meta.get("best_val")
    if bv is not None:
        bits.append(f"best_val={bv:.4f}")
    return "  ".join(bits)


def run_chat(model: GPT, tok, device: torch.device, args) -> None:
    """
    Multi-turn chat through the shared chat template.

    Two things matter here and nowhere else:
      * the prompt is built by src/template.build_prompt, so the model sees the
        EXACT token sequence it was fine-tuned on. Feeding a bare string to an
        SFT model produces story continuations, because that is an input it has
        never seen.
      * generation stops at <|eos|> -- the token SFT taught the model to emit at
        the end of a reply.
    """
    ctx = model.cfg.ctx_len
    history: list[dict] = []

    print("\n  Chat mode -- replies come from the fine-tuned model.")
    print("  Commands: :q quit   :reset clear history   :history show turns   :settings")
    print(f"  system  : {args.system}")
    print(f"  context : {ctx} tokens\n")

    while True:
        try:
            line = input("  you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line == ":reset":
            history.clear()
            print("  (history cleared)")
            continue
        if line == ":history":
            for m in history:
                print(f"    {m['role']:>9}: {m['content'][:68]}")
            if not history:
                print("    (empty)")
            continue
        if line == ":settings":
            print(
                f"    temperature={args.temperature} top_p={args.top_p} "
                f"repetition_penalty={args.repetition_penalty} "
                f"max_new_tokens={args.max_new_tokens}"
            )
            print(f"    system={args.system!r}")
            continue

        history.append({"role": "user", "content": line})

        # Trim the oldest COMPLETE turns until the prompt fits the context.
        # Trimming mid-turn would leave a half conversation, which is a shape
        # the model never saw in training.
        while len(history) > 1:
            probe = tok.encode(
                build_prompt(history, system=args.system, add_generation_prompt=True)
            )
            if len(probe) + args.max_new_tokens <= ctx:
                break
            history.pop(0)
            if history and history[0]["role"] == "assistant":
                history.pop(0)

        ids = tok.encode(
            build_prompt(history, system=args.system, add_generation_prompt=True)
        )
        if len(ids) + args.max_new_tokens > ctx:
            ids = ids[-(ctx - args.max_new_tokens) :]  # single very long turn

        x = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.time()
        out = model.generate(
            x,
            args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_id=EOS_ID,
        )
        new_ids = out[0].tolist()[len(ids) :]
        reply = tok.decode(new_ids, skip_special_tokens=True).strip()
        dt = time.time() - t0

        if args.show_ids:
            print(f"    ids: {new_ids[:30]}{'...' if len(new_ids) > 30 else ''}")
        print(f"  bot > {reply}")
        print(
            f"        ({len(new_ids)} tok, {dt:.2f}s, "
            + ("stopped at EOS)" if EOS_ID in new_ids else "hit max_new_tokens -- "
               "it may not have finished)")
        )

        history.append({"role": "assistant", "content": reply})

    print("  bye")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default="runs/tiny_smoke/best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/tiny_tokenizer.json")
    ap.add_argument("--prompt", action="append", default=[],
                    help="repeatable; defaults to a few built-in prompts")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--interactive", action="store_true",
                    help="free continuation: the model appends to whatever you type. "
                         "This is NOT chat -- use --chat for an SFT model.")
    ap.add_argument("--chat", action="store_true",
                    help="wrap input in the chat template and print only the reply. "
                         "Needs a checkpoint fine-tuned with src.train_sft.")
    ap.add_argument("--system", default=DEFAULT_SYSTEM,
                    help="system prompt used in --chat mode")
    ap.add_argument("--max-history-turns", type=int, default=4,
                    help="in --chat, drop the oldest user/assistant pairs once the "
                         "conversation no longer fits the context window")
    ap.add_argument("--show-ids", action="store_true")
    ap.add_argument("--stop-at-eos", action="store_true", default=True)
    args = ap.parse_args(argv)

    # ---- device ---------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---- tokenizer ------------------------------------------------------
    from src.tokenizer import BPETokenizer

    tk_path = resolve(args.tokenizer)
    if not tk_path.exists():
        raise SystemExit(f"tokenizer not found: {describe_missing(args.tokenizer)}")
    tok = BPETokenizer.load(tk_path)

    # ---- model ----------------------------------------------------------
    ckpt_path = resolve(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(
            f"checkpoint not found: {describe_missing(args.ckpt)}\n"
            "    train one first:\n"
            "      run.cmd -m src.train_pretrain --preset tiny --max-iters 600 "
            "--out-dir runs/tiny_smoke --micro-batch 32 --grad-accum 2"
        )
    model, meta = load_model(ckpt_path, device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"  checkpoint : {describe_ckpt(ckpt_path, meta)}")
    print(f"  tokenizer  : {tk_path.name}  (vocab {tok.vocab_size:,})")
    print(f"  model      : {model.cfg.summary()}")
    print(f"  device     : {device}")
    print("-" * 72)

    prompts = args.prompt or [
        "Once upon a time",
        "One day, a little girl",
        "The cat",
        "Lily and Tom",
    ]

    def run_one(prompt: str, idx_sample: int) -> None:
        ids = tok.encode(prompt)
        if not ids:
            print("  (empty prompt)")
            return
        if len(ids) >= model.cfg.ctx_len:
            ids = ids[-model.cfg.ctx_len + 1 :]

        x = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.time()
        out = model.generate(
            x,
            args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_id=EOS_ID if args.stop_at_eos else None,
        )
        dt = time.time() - t0
        new_ids = out[0].tolist()[len(ids) :]
        text = tok.decode(new_ids)

        print(f"\n  [{prompt!r}]  (sample {idx_sample + 1})")
        if args.show_ids:
            print(f"    ids : {new_ids[:40]}{'...' if len(new_ids) > 40 else ''}")
        print(f"    ---")
        # indented, wrapped-ish output
        for line in text.replace("\n", " ").split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
        print(f"    ---")
        print(
            f"    {len(new_ids)} tokens in {dt:.2f}s "
            f"({len(new_ids)/max(dt,1e-6):.0f} tok/s)"
            + ("  [stopped at EOS]" if args.stop_at_eos and EOS_ID in new_ids else "")
        )

    if args.chat:
        run_chat(model, tok, device, args)
        return 0

    if args.interactive:
        print("\n  Interactive CONTINUATION mode. Type a prompt and press Enter.")
        print("  The model appends to your text. This is not chat -- it will not")
        print("  answer questions, it will keep writing. Use --chat for that.\n"
              "  Type :q to quit, :settings to see the sampling parameters.\n")
        while True:
            try:
                line = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line in (":q", ":quit", ":exit"):
                break
            if line == ":settings":
                print(
                    f"    temperature={args.temperature} top_p={args.top_p} "
                    f"repetition_penalty={args.repetition_penalty} "
                    f"max_new_tokens={args.max_new_tokens}"
                )
                continue
            if not line:
                line = "Once upon a time"
            run_one(line, 0)
        print("  bye")
        return 0

    for i, p in enumerate(prompts):
        for s in range(args.num_samples):
            run_one(p, s)

    print(
        f"\n  note: temperature={args.temperature} "
        f"({'greedy/deterministic' if args.temperature <= 0 else 'sampling'}), "
        f"top_p={args.top_p}, repetition_penalty={args.repetition_penalty}"
    )
    if args.temperature > 0:
        print("  run with --temperature 0 to see the model's single best guess.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
