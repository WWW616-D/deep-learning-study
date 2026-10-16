"""
Model and training configuration.

Two presets:

  TINY : ~6.3M params  -- smoke test. Validates the ENTIRE pipeline in minutes,
                          on tiny data, so that bugs surface before we spend
                          hours on a real run.
  BASE : ~31.5M params -- the real run.

Why these numbers (the reasoning matters more than the values):

  * head_dim is pinned to 64. d_model must therefore be a multiple of n_head,
    and both 256/4 and 384/6 give a clean 64.
  * Per-layer cost is about 16 * d_model^2:
        attention (q,k,v,o) = 4*d^2
        SwiGLU MLP          = 3*d*d_ff, and d_ff = 4*d gives 12*d^2
  * Embedding share must stay under ~45% of total params:
        share = V*d / (V*d + 16*L*d^2)
    Shrinking d_model while keeping a large vocab is a trap -- the model
    becomes mostly a lookup table with almost no compute per token.
  * Small models want DEPTH over WIDTH. For reference, SmolLM2-135M uses
    d_model=576 with 30 layers -- narrower than GPT-2 small (768) but far
    deeper. So we spend our parameter budget on layers, not width.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Architecture of the transformer itself."""

    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int
    ctx_len: int

    rope_theta: float = 10_000.0
    qk_norm: bool = True          # RMSNorm on q/k per head; cheap stabiliser
    tie_embeddings: bool = True   # share input embedding and output projection
    dropout: float = 0.0          # 0 for pretraining; raise for SFT on small data

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        head_dim = self.d_model // self.n_head
        if head_dim not in (32, 64, 128):
            # Not fatal, but almost always a mistake.
            import warnings

            warnings.warn(
                f"head_dim={head_dim} is unusual; 64 or 128 is the healthy range",
                stacklevel=2,
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_params_non_embedding(self) -> int:
        """Parameters that actually do compute, i.e. excluding the lookup table."""
        per_layer_attn = 4 * self.d_model * self.d_model
        per_layer_mlp = 3 * self.d_model * self.d_ff
        per_layer_norm = 2 * self.d_model  # two RMSNorms per block
        # QK-Norm is applied per head, so each RMSNorm holds head_dim weights
        # (not d_model). Getting this wrong silently mis-states the model size.
        per_layer_qk = 2 * self.head_dim if self.qk_norm else 0
        per_layer = per_layer_attn + per_layer_mlp + per_layer_norm + per_layer_qk
        return self.n_layer * per_layer + self.d_model  # + final norm

    @property
    def n_params_embedding(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def n_params(self) -> int:
        """Total parameter count (input embedding counted once even when tied)."""
        return self.n_params_embedding + self.n_params_non_embedding

    @property
    def embedding_share(self) -> float:
        return self.n_params_embedding / self.n_params

    def summary(self) -> str:
        return (
            f"vocab={self.vocab_size} d_model={self.d_model} n_layer={self.n_layer} "
            f"n_head={self.n_head} head_dim={self.head_dim} d_ff={self.d_ff} "
            f"ctx={self.ctx_len} | params={self.n_params/1e6:.2f}M "
            f"(emb {self.embedding_share*100:.0f}%)"
        )


@dataclass
class TrainConfig:
    """Optimisation settings."""

    # --- data ---
    data_dir: str = "data/processed"
    seq_len: int = 1024
    # --- batching: effective tokens/step = micro_batch * grad_accum * seq_len ---
    micro_batch: int = 8
    grad_accum: int = 8
    # --- optimisation ---
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 200
    max_iters: int = 6000
    # --- runtime ---
    dtype: str = "bf16"          # bf16 | fp16 | fp32
    grad_checkpointing: bool = False
    compile: bool = False        # torch.compile; slow first step, faster after
    device: str = "auto"
    seed: int = 1337
    # --- logging / checkpointing ---
    out_dir: str = "runs/base"
    log_interval: int = 10
    eval_interval: int = 250
    eval_iters: int = 100
    save_interval: int = 500     # write a resumable checkpoint every N steps
    keep_last_n_ckpts: int = 3   # rotate step_*.pt, keeping only the newest N
    sample_interval: int = 500   # generate a short sample so you can SEE progress
    sample_prompt: str = "Once upon a time"

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.seq_len

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.max_iters


# --------------------------------------------------------------------------
#  Presets
# --------------------------------------------------------------------------

TINY = ModelConfig(
    vocab_size=8192,
    d_model=256,
    n_layer=4,
    n_head=4,      # head_dim = 64
    d_ff=1024,
    ctx_len=256,
)

BASE = ModelConfig(
    vocab_size=32768,
    d_model=384,
    n_layer=8,
    n_head=6,      # head_dim = 64
    d_ff=1536,
    ctx_len=1024,
)

PRESETS: Dict[str, ModelConfig] = {"tiny": TINY, "base": BASE}


def get_preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]


def as_dict(cfg: Any) -> Dict[str, Any]:
    return asdict(cfg)


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        print(f"[{name}] {cfg.summary()}")
        print(
            f"        non-embedding params = {cfg.n_params_non_embedding/1e6:.2f}M, "
            f"per-token compute is spread over {cfg.n_layer} layers"
        )
