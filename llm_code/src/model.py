"""
Minimal decoder-only GPT, written from scratch in plain PyTorch.

Architecture, and why each piece was chosen for a ~30M-parameter model:

  * Pre-Norm + RMSNorm
        Normalising before each sub-layer is what makes deep stacks trainable.
        RMSNorm (Zhang & Sennrich 2019) drops the mean subtraction and the
        bias, so it is cheaper than LayerNorm and works at least as well.
  * RoPE (rotary position embedding)
        Encodes position by rotating q/k. No learned position table, so no
        extra parameters, and relative offsets generalise sensibly.
  * SwiGLU MLP
        down(silu(gate(x)) * up(x)) -- better quality per parameter than a
        plain GELU MLP, at the cost of one extra matrix.
  * QK-Norm
        RMSNorm applied to q and k per head. Nearly free, and noticeably
        stabilises small models, whose gradients are more prone to spikes.
  * Tied embeddings
        The input embedding matrix doubles as the output projection. Saves
        V*d params -- about 40% of this model -- and generally helps small
        models rather than hurting them.
  * No biases anywhere
        Modern practice; saves parameters and loses nothing measurable.
  * KV cache
        Required for generation that is not embarrassingly slow.

Deliberately NOT included:
  * GQA. It shrinks the KV cache, but we decode with batch=1 and ctx<=1024,
    so the cache is a few MB. Not worth the extra complexity at this scale.
  * MoE, sliding-window/local attention, attention sinks, muP.
    All real techniques, none of them useful at 30M params and ctx=1024.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig, TrainConfig


# ==========================================================================
#  Building blocks
# ==========================================================================


class RMSNorm(nn.Module):
    """Root-mean-square normalisation, computed in float32 for stability."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class RoPE(nn.Module):
    """
    Rotary position embedding.

    The cache is built once for max_seq_len. Frequencies are stored as
    [f0..f_{D/2-1}, f0..f_{D/2-1}] so that _rotate_half() pairs (i, i+D/2),
    which is the layout used by Llama-style implementations.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)          # [S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [S, D]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = q.size(-2)
        if start_pos + t > self.max_seq_len:
            raise ValueError(
                f"RoPE cache is built for {self.max_seq_len} positions but was "
                f"asked for positions {start_pos}..{start_pos + t - 1}"
            )
        cos = self.cos_cached[start_pos : start_pos + t].to(q.dtype)[None, None]
        sin = self.sin_cached[start_pos : start_pos + t].to(q.dtype)[None, None]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


class LayerKVCache:
    """Append-only key/value store for one attention layer."""

    __slots__ = ("k", "v")

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.size(2)

    def reset(self) -> None:
        self.k = self.v = None


def new_kv_cache(n_layer: int) -> List[LayerKVCache]:
    return [LayerKVCache() for _ in range(n_layer)]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None

        self.rope = RoPE(self.head_dim, cfg.ctx_len, cfg.rope_theta)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[LayerKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)  # type: ignore[misc]

        q, k = self.rope(q, k, start_pos)

        if cache is not None:
            k, v = cache.append(k, v)

        # is_causal=True builds a lower-triangular mask sized q_len x k_len.
        # That is only correct when q and k have equal length (prefill).
        # During cached decoding q_len == 1 while k_len == full context, and a
        # causal mask would wrongly allow the single query to see only key 0.
        is_causal = q.size(2) == k.size(2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.resid_dropout(self.o_proj(y))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[LayerKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cache, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x


# ==========================================================================
#  The model
# ==========================================================================


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)

        # Weight tying is implemented inside forward() by reusing tok_emb.weight,
        # so no separate module is created when tied.
        self.lm_head: Optional[nn.Linear] = (
            None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        )

        self.apply(self._init_weights)
        # GPT-2 style scaled init for the projections that write into the
        # residual stream: their contribution accumulates over 2*n_layer adds,
        # so they are scaled down to keep the residual variance stable.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.grad_checkpointing = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[List[LayerKVCache]] = None,
        start_pos: int = 0,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            if self.grad_checkpointing and self.training:
                # Recompute activations in the backward pass instead of storing
                # them. Slower, but the difference between OOM and not.
                x = torch.utils.checkpoint.checkpoint(
                    block, x, layer_cache, start_pos, use_reentrant=False
                )
            else:
                x = block(x, layer_cache, start_pos)

        x = self.norm_f(x)

        if self.cfg.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)  # type: ignore[misc]

        if targets is None:
            return logits, None

        loss = compute_loss(logits, targets, loss_mask)
        return logits, loss

    # ------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive sampling.

        `idx` is the prompt, shape [B, T]. Returns the prompt with the newly
        generated tokens appended. Setting use_cache=False recomputes the whole
        context every step; the two paths must produce identical output for the
        same seed, and tests/test_kv_cache.py asserts exactly that.
        """
        was_training = self.training
        self.eval()
        ctx = self.cfg.ctx_len

        cache = new_kv_cache(self.cfg.n_layer) if use_cache else None
        fed = 0  # how many positions the model has already consumed

        for _ in range(max_new_tokens):
            if use_cache:
                cur = idx[:, fed:]
                pos = fed
            else:
                cur = idx[:, -ctx:]
                pos = 0

            # Stop BEFORE feeding past the context window: the RoPE cache only
            # covers ctx positions, and a position beyond it would either raise
            # or silently wrap. Checking here (rather than after appending)
            # keeps the returned sequence within ctx.
            if cur.size(1) == 0 or pos + cur.size(1) > ctx:
                break

            logits, _ = self(cur, cache=cache, start_pos=pos)
            logits = logits[:, -1, :]
            fed += cur.size(1)

            logits = _apply_sampling(
                logits, idx, temperature, top_p, repetition_penalty
            )
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and bool((next_id == eos_id).all()):
                break

        if was_training:
            self.train()
        return idx

    # ---------------------------------------------------------------- helpers

    def configure_optimizers(self, tcfg: TrainConfig) -> torch.optim.Optimizer:
        """
        AdamW with decay applied only to matrices.

        Norm weights and biases are 1-D and are left undecayed -- decaying them
        toward zero is not meaningful and measurably hurts.
        """
        named = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in named.values() if p.dim() >= 2]
        no_decay = [p for p in named.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups, lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2), eps=1e-8
        )

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def estimate_flops_per_token(self) -> float:
        """Rough forward+backward cost, the usual 6*N*D rule."""
        n = self.num_params(non_embedding=True)
        return 6.0 * n

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "GPT":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**ckpt["model_config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["model"])
        return model

    def embedding_norms(self) -> torch.Tensor:
        """Per-token L2 norm of the input embedding, shape [V]."""
        return self.tok_emb.weight.detach().norm(dim=1)


# ==========================================================================
#  Loss
# ==========================================================================


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross-entropy between predicted next-token logits and targets.

    logits  : [B, T, V]
    targets : [B, T]      (already shifted -- position t predicts targets[t])
    loss_mask: [B, T]     1.0 to count a position, 0.0 to ignore it.
              Used by SFT to train only on assistant turns.

    A logit of 0.0 in bf16 padding positions is harmless because the masked
    entries are removed before the reduction.
    """
    if loss_mask is None:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(flat_logits.dtype)

    n_active = flat_mask.sum()
    if n_active.item() == 0:
        # No supervised position in this batch. Return a zero that still
        # carries gradients so DDP/backward does not blow up.
        return flat_logits.sum() * 0.0

    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (per_token * flat_mask).sum() / n_active


def _apply_sampling(
    logits: torch.Tensor,
    prev_ids: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty, temperature and nucleus sampling to logits."""
    logits = logits.float()

    if repetition_penalty != 1.0:
        for b in range(logits.size(0)):
            seen = torch.unique(prev_ids[b])
            vals = logits[b, seen]
            # Positive logits are divided, negative ones multiplied: both
            # actions push the token's probability down.
            logits[b, seen] = torch.where(
                vals > 0, vals / repetition_penalty, vals * repetition_penalty
            )

    if temperature <= 0.0:  # greedy
        best = logits.argmax(dim=-1, keepdim=True)
        out = torch.full_like(logits, float("-inf"))
        return out.scatter_(-1, best, 0.0)

    logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep the smallest set whose cumulative probability reaches top_p;
        # `- probs` keeps the token that crosses the threshold.
        cut = cumulative - probs > top_p
        sorted_logits = sorted_logits.masked_fill(cut, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    return logits
