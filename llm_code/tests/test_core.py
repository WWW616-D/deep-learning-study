"""
Dependency-free self-tests for the core components.

Run from the project root:

    python tests/test_core.py

Every check prints PASS/FAIL with the actual number it measured. Nothing here
"looks about right" -- if something is wrong we want it to be loud, because the
failures that matter in LLM training are silent ones.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import BASE, TINY, ModelConfig, TrainConfig  # noqa: E402
from src.model import GPT, compute_loss, new_kv_cache  # noqa: E402
from src import template as tmpl  # noqa: E402

PASS, FAIL = "[PASS]", "[FAIL]"
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {PASS if ok else FAIL} {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ==========================================================================
#  1. Environment
# ==========================================================================


def test_environment() -> None:
    section("1. Environment")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  torch       : {torch.__version__}")
    print(f"  cuda build  : {torch.version.cuda}")
    print(f"  cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"  gpu         : {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
        print(f"  vram        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  gpu         : none -- CPU only.")

    cpu_only = torch.version.cuda is None
    if cpu_only:
        print(
            "\n  NOTE: this is a CPU-only torch build. Training will work but be\n"
            "        50-100x slower. Install the CUDA build before the real run:\n"
            "        pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )


# ==========================================================================
#  2. Parameter accounting
# ==========================================================================


def test_param_counts() -> None:
    section("2. Parameter accounting (config estimate vs. real model)")
    for name, cfg in (("tiny", TINY), ("base", BASE)):
        model = GPT(cfg)
        actual = sum(p.numel() for p in model.parameters())
        predicted = cfg.n_params
        diff = abs(actual - predicted)
        print(f"\n  [{name}] {cfg.summary()}")
        print(
            f"         predicted {predicted/1e6:8.3f}M | actual {actual/1e6:8.3f}M "
            f"| diff {diff} params"
        )
        check(
            f"{name}: estimate matches real model",
            diff == 0,
            "exact" if diff == 0 else f"off by {diff}",
        )
        emb = cfg.n_params_embedding
        check(
            f"{name}: embedding share <= 47%",
            cfg.embedding_share <= 0.47,
            f"{cfg.embedding_share*100:.1f}%",
        )
        check(
            f"{name}: head_dim == 64",
            cfg.head_dim == 64,
            f"head_dim={cfg.head_dim}",
        )
        del model


# ==========================================================================
#  3. Chat template
# ==========================================================================


def test_template() -> None:
    section("3. Chat template (must be byte-identical between train and inference)")

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "how are you"},
    ]
    prompt = tmpl.build_prompt(msgs, system="sys", add_generation_prompt=True)
    expected = (
        "<|bos|><|system|>sys<|eos|><|user|>hi<|eos|>"
        "<|assistant|>hello!<|eos|><|user|>how are you<|eos|><|assistant|>"
    )
    check("build_prompt produces the expected string", prompt == expected)
    if prompt != expected:
        print(f"         got:      {prompt!r}")
        print(f"         expected: {expected!r}")

    # Round-trip: parse -> rebuild must be stable.
    parsed = tmpl.parse_conversation(
        tmpl.build_prompt(msgs, system="sys", add_generation_prompt=False)
    )
    check(
        "parse_conversation round-trips",
        parsed[0]["role"] == "system" and parsed[-1]["content"] == "how are you",
        f"{len(parsed)} messages",
    )

    # A system message inside `messages` must be rejected loudly.
    try:
        tmpl.build_prompt([{"role": "system", "content": "x"}])
        check("nested system message is rejected", False, "no exception raised")
    except ValueError:
        check("nested system message is rejected", True)

    # Content stripping must be consistent, otherwise train/infer drift.
    a = tmpl.build_prompt([{"role": "user", "content": "  hi  "}], system="s")
    b = tmpl.build_prompt([{"role": "user", "content": "hi"}], system="s")
    check("whitespace is normalised consistently", a == b)

    check("6 special tokens defined", len(tmpl.SPECIAL_TOKENS) == 6, str(tmpl.SPECIAL_TOKENS))


# ==========================================================================
#  4. Forward pass and loss
# ==========================================================================


def test_forward_and_loss() -> None:
    section("4. Forward pass and loss")

    cfg = TINY
    model = GPT(cfg).eval()
    B, T = 2, min(32, cfg.ctx_len)

    # Shifted targets: position t must predict token t+1. This mirrors real
    # pretraining, and it matters for the sanity check below. If instead the
    # target were the token AT position t, the model could partially read the
    # answer out of its own embedding (the tied weights make the input row and
    # the output prototype the same vector), so the loss would sit noticeably
    # below ln(V) even at initialisation.
    seq = torch.randint(0, cfg.vocab_size, (B, T + 1))
    idx = seq[:, :-1].contiguous()
    targets = seq[:, 1:].contiguous()

    with torch.no_grad():
        logits, loss = model(idx, targets=targets)
    check("logits shape is [B,T,V]", tuple(logits.shape) == (B, T, cfg.vocab_size),
          str(tuple(logits.shape)))

    # A freshly initialised model knows nothing, so the loss must sit near
    # ln(V). Far below means something leaks; far above means the init is bad.
    expected = torch.log(torch.tensor(float(cfg.vocab_size))).item()
    check(
        "initial loss is close to ln(vocab_size)",
        abs(loss.item() - expected) < 0.05 * expected,
        f"loss={loss.item():.4f}  ln(V)={expected:.4f}",
    )

    # Document the self-copy shortcut rather than pretending it is not there.
    with torch.no_grad():
        _, copy_loss = model(idx, targets=idx)
    print(
        f"         (targets==idx gives {copy_loss.item():.4f}, which is below "
        f"ln(V)={expected:.4f}: the tied-weight self-copy shortcut)"
    )
    check(
        "self-target loss is below shifted-target loss (shortcut documented)",
        copy_loss.item() < loss.item(),
    )

    # --- loss mask -------------------------------------------------------
    mask = torch.zeros(B, T)
    mask[:, ::2] = 1.0
    with torch.no_grad():
        _, masked_loss = model(idx, targets=targets, loss_mask=mask)

    flat_logits = logits.reshape(-1, cfg.vocab_size)
    per_token = torch.nn.functional.cross_entropy(
        flat_logits, targets.reshape(-1), reduction="none"
    )
    manual = per_token.reshape(B, T)[mask.bool()].mean()
    check(
        "loss mask matches a manual reduction",
        torch.allclose(masked_loss, manual, atol=1e-5),
        f"masked={masked_loss.item():.6f}  manual={manual.item():.6f}",
    )

    zero_mask = torch.zeros(B, T)
    with torch.no_grad():
        _, empty_loss = model(idx, targets=targets, loss_mask=zero_mask)
    check("all-zero mask returns 0 without NaN", empty_loss.item() == 0.0)


# ==========================================================================
#  5. KV cache equivalence  -- the classic silent bug
# ==========================================================================


def test_kv_cache() -> None:
    section("5. KV cache equivalence (incremental decoding must match full forward)")

    cfg = TINY
    model = GPT(cfg).eval()
    T = min(48, cfg.ctx_len)
    idx = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full_logits, _ = model(idx)  # no cache: one shot

        cache = new_kv_cache(cfg.n_layer)
        step_logits = []
        for t in range(T):
            lg, _ = model(idx[:, t : t + 1], cache=cache, start_pos=t)
            step_logits.append(lg[:, 0, :])
        step_logits = torch.stack(step_logits, dim=1)

    max_diff = (full_logits - step_logits).abs().max().item()
    check(
        "cached and uncached logits agree",
        max_diff < 1e-4,
        f"max |diff| = {max_diff:.3e}",
    )

    # Greedy generation through both paths must produce identical tokens.
    torch.manual_seed(0)
    out_cached = model.generate(idx[:, :4], 20, temperature=0.0, use_cache=True)
    torch.manual_seed(0)
    out_uncached = model.generate(idx[:, :4], 20, temperature=0.0, use_cache=False)
    check(
        "greedy generation identical with and without cache",
        torch.equal(out_cached, out_uncached),
        f"{out_cached.shape[1]} tokens",
    )


# ==========================================================================
#  6. Sampling behaviour
# ==========================================================================


def test_sampling() -> None:
    section("6. Sampling controls")

    cfg = TINY
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))

    torch.manual_seed(0)
    greedy = model.generate(prompt, 12, temperature=0.0)
    torch.manual_seed(1)
    greedy2 = model.generate(prompt, 12, temperature=0.0)
    check("temperature=0 is deterministic", torch.equal(greedy, greedy2))

    torch.manual_seed(0)
    sampled = model.generate(prompt, 12, temperature=1.0, top_p=0.9)
    check("sampled output differs from greedy", not torch.equal(greedy, sampled))

    # EOS must stop generation early.
    cfg2 = ModelConfig(
        vocab_size=256, d_model=64, n_layer=1, n_head=1, d_ff=128, ctx_len=64
    )
    m2 = GPT(cfg2).eval()
    prompt = torch.zeros(1, 1, dtype=torch.long)

    # A randomly initialised model will not spontaneously emit an arbitrary EOS
    # id, so first discover which token greedy decoding actually produces, then
    # use THAT as EOS and assert the loop really halts on it.
    torch.manual_seed(0)
    probe = m2.generate(prompt, 3, temperature=0.0)
    forced_eos = probe[0, -1].item()
    torch.manual_seed(0)
    out = m2.generate(prompt, 40, temperature=0.0, eos_id=forced_eos)
    check(
        "eos_id stops generation early",
        out.shape[1] == prompt.shape[1] + 1,
        f"emitted {out.shape[1] - prompt.shape[1]} token(s), eos_id={forced_eos}",
    )

    # Context window: generation stops as soon as the window is full. The
    # returned sequence is at most ctx_len + 1 tokens -- the extra one is the
    # model's final output, which filled the window and was never fed back.
    torch.manual_seed(0)
    out = m2.generate(prompt, 200, temperature=0.0)
    check(
        "generation respects the context window",
        out.shape[1] <= cfg2.ctx_len + 1,
        f"len={out.shape[1]} ctx={cfg2.ctx_len}",
    )

    # A prompt that already fills the window leaves room for exactly one token.
    torch.manual_seed(0)
    full_prompt = torch.zeros(1, cfg2.ctx_len, dtype=torch.long)
    out = m2.generate(full_prompt, 50, temperature=0.0)
    check(
        "a full-length prompt yields exactly one more token",
        out.shape[1] == cfg2.ctx_len + 1,
        f"len={out.shape[1]}",
    )


# ==========================================================================
#  7. Single-batch overfit  -- proves the whole training loop is wired up
# ==========================================================================


def test_single_batch_overfit() -> None:
    section("7. Single-batch overfit (the strongest smoke test there is)")
    print("  If a model cannot memorise ONE batch, something in the training")
    print("  loop is broken -- and no amount of real data will fix that.\n")

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=512, d_model=128, n_layer=2, n_head=2, d_ff=512, ctx_len=64
    )
    model = GPT(cfg)
    tcfg = TrainConfig(lr=3e-3, weight_decay=0.0, warmup_iters=10, grad_clip=1.0)
    opt = model.configure_optimizers(tcfg)

    B, T = 4, 32
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    model.train()
    losses = []
    t0 = time.time()
    for step in range(200):
        _, loss = model(idx, targets=targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % 40 == 0 or step == 199:
            print(f"         step {step:4d}   loss {loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"         ({elapsed:.1f}s for 200 steps on CPU)")

    check(
        "loss collapses toward zero on a memorisable batch",
        losses[-1] < 0.05,
        f"{losses[0]:.3f} -> {losses[-1]:.4f}",
    )
    check(
        "loss decreased monotonically in aggregate",
        losses[-1] < losses[0] * 0.05,
    )


# ==========================================================================
#  8. Gradient flow into the embedding
# ==========================================================================


def test_embedding_gradients() -> None:
    section("8. Gradient flow into the embedding table")

    cfg = ModelConfig(
        vocab_size=1000, d_model=64, n_layer=1, n_head=1, d_ff=128, ctx_len=32
    )
    model = GPT(cfg)
    model.train()

    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(idx, targets=idx)
    loss.backward()

    grad = model.tok_emb.weight.grad
    check("embedding receives a gradient", grad is not None)

    rows_with_grad = (grad.abs().sum(dim=1) > 0).sum().item()
    present = torch.unique(idx).numel()
    print(f"         tokens in batch         : {present}")
    print(f"         embedding rows with grad: {rows_with_grad}")
    print(
        "\n         This is the sparsity point: with separate (untied) embeddings\n"
        "         only the rows that appear get a gradient. With TIED embeddings\n"
        "         the output side also touches every row, but with a magnitude\n"
        "         proportional to its softmax probability -- so rare tokens stay\n"
        "         badly trained either way."
    )

    norms = model.embedding_norms()
    check("embedding norms are finite and positive",
          torch.isfinite(norms).all() and (norms > 0).all(),
          f"mean={norms.mean():.4f} max={norms.max():.4f}")


# ==========================================================================
#  9. SFT: template consistency and loss mask
# ==========================================================================


def test_sft_pipeline() -> None:
    section("9. SFT: template consistency and loss mask (the silent-bug zone)")

    tk_path = ROOT / "data/tokenizer/tiny_tokenizer.json"
    if not tk_path.exists():
        print(f"  [SKIP] {tk_path} does not exist -- train a tokenizer first")
        return

    from src.sft_data import SFTDataset, build_example
    from src.tokenizer import BPETokenizer

    tok = BPETokenizer.load(tk_path)

    messages = [
        {"role": "user", "content": "What is an adversarial example?"},
        {"role": "assistant", "content": "An input changed so a model misclassifies it."},
        {"role": "user", "content": "How is it evaluated?"},
        {"role": "assistant", "content": "With strong adaptive attacks and worst-case accuracy."},
    ]
    system = "You are a technical assistant."

    ids_inc, mask = build_example(tok, messages, system)
    ids_tpl = tok.encode(tmpl.build_prompt(messages, system=system))

    # ---- THE critical check -------------------------------------------
    # If these differ, training sees a different token sequence than inference
    # does. There is no error anywhere; the model just looks bad.
    check(
        "build_example == build_prompt + encode  (train/infer identical)",
        ids_inc == ids_tpl,
        f"{len(ids_inc)} vs {len(ids_tpl)} tokens",
    )
    if ids_inc != ids_tpl:
        for i, (a, b) in enumerate(zip(ids_inc, ids_tpl)):
            if a != b:
                print(f"         first difference at index {i}: {a} vs {b}")
                break

    # ---- mask covers assistant turns only ------------------------------
    supervised = tok.decode(
        [i for i, m in zip(ids_inc, mask) if m == 1], skip_special_tokens=False
    )
    check(
        "assistant content IS supervised",
        "An input changed" in supervised and "worst-case accuracy" in supervised,
        repr(supervised[:48]),
    )
    check(
        "user content is NOT supervised",
        "What is an adversarial example" not in supervised
        and "How is it evaluated" not in supervised,
    )
    check(
        "system content is NOT supervised",
        "You are a technical assistant" not in supervised,
    )

    # ---- exactly the assistant-closing EOS is supervised ---------------
    n_eos_sup = sum(1 for i, m in enumerate(mask) if m == 1 and ids_inc[i] == tmpl.EOS_ID)
    n_eos_all = sum(1 for i in ids_inc if i == tmpl.EOS_ID)
    check(
        "only assistant-closing EOS is supervised (this teaches stopping)",
        n_eos_sup == 2 and n_eos_all == 5,
        f"{n_eos_sup} supervised of {n_eos_all} total EOS",
    )

    # ---- role tokens are never predicted -------------------------------
    role_ids = {tmpl.SYSTEM_ID, tmpl.USER_ID, tmpl.ASSISTANT_ID, tmpl.BOS_ID}
    check(
        "role tokens and BOS are never supervised",
        all(m == 0 for i, m in enumerate(mask) if ids_inc[i] in role_ids),
    )

    # ---- dataset: shift alignment and padding --------------------------
    max_len = 160
    assert len(ids_inc) < max_len, "test conversation grew past max_len"
    ds = SFTDataset([{"system": system, "messages": messages}], tok, max_len=max_len,
                    verbose=False)
    x, y, m = ds[0]
    check("x / y / mask all have the same length",
          x.shape == y.shape == m.shape == (max_len - 1,), str(tuple(x.shape)))

    padded = torch.tensor(ids_inc + [ds.pad_id] * (max_len - len(ids_inc)))
    check("targets are the inputs shifted by exactly one",
          torch.equal(torch.cat([x, y[-1:]]), padded))
    check("inputs equal the padded sequence minus its last token",
          torch.equal(x, padded[:-1]))

    pad_positions = (x == ds.pad_id).nonzero().flatten()
    check("padding is never supervised",
          bool((m[pad_positions] == 0).all()) if len(pad_positions) else True,
          f"{len(pad_positions)} padded positions")

    # mask is aligned to the TARGETS, not the inputs: m[t] supervises ids[t+1]
    check("mask is shifted to match the targets",
          torch.equal(m, torch.tensor(mask[1:] + [0] * (max_len - len(mask)))))

    # ---- masked loss equals a manual reduction -------------------------
    cfg = TINY
    model = GPT(cfg).eval()
    V = min(cfg.vocab_size, tok.vocab_size)
    xx = torch.randint(0, V, (2, 32))
    yy = torch.randint(0, V, (2, 32))
    mm = torch.zeros(2, 32)
    mm[:, 8:20] = 1.0
    with torch.no_grad():
        logits, _ = model(xx)
        masked = compute_loss(logits, yy, mm)
        manual = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yy.reshape(-1), reduction="none"
        ).reshape(2, 32)[mm.bool()].mean()
    check("masked loss matches a manual reduction",
          torch.allclose(masked, manual, atol=1e-5),
          f"{masked.item():.6f} vs {manual.item():.6f}")


# ==========================================================================

def main() -> int:
    print("llm_code -- core self-tests")
    test_environment()
    test_param_counts()
    test_template()
    test_forward_and_loss()
    test_kv_cache()
    test_sampling()
    test_single_batch_overfit()
    test_embedding_gradients()
    test_sft_pipeline()

    section("Summary")
    if _failures:
        print(f"  {len(_failures)} check(s) FAILED:")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
