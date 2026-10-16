"""
Chat template -- THE single source of truth for prompt assembly.

Both SFT training and inference must build prompts through this module.
A mismatch here (one missing space, one different newline) silently destroys
model quality: the model sees a token sequence at inference time that it never
saw during training, and the failure looks like "the model is just bad" rather
than like a bug.

Therefore:
  * build_prompt() is the ONLY place where prompt strings are assembled.
  * training-script and inference-script both import it -- never hand-roll it.
  * tests/test_template.py asserts the exact token output.

Wire format
-----------
    <|bos|><|system|>{system}<|eos|><|user|>{u1}<|eos|><|assistant|>{a1}<|eos|>...

  * Every message is wrapped as  <|role|>content<|eos|>
  * The whole conversation is prefixed with a single <|bos|>
  * During generation the prompt ends right after <|assistant|>, and the model
    is expected to emit its reply followed by <|eos|>
  * Multi-turn: keep appending  <|user|>...<|eos|><|assistant|>...<|eos|>

Why role tokens instead of plain "User:"/"Assistant:" text
---------------------------------------------------------
Role tokens are single, atomic token ids. Plain text markers get chopped into
several BPE fragments, which costs context budget and gives the model a fuzzier
signal about where the turn boundary is. At 30M parameters, every token of
context counts.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Sequence, TypedDict

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


# Order matters: these are the first ids reserved in the tokenizer.
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

SPECIAL_TOKENS: List[str] = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
]

# Convenience id constants for the indexing scheme we always use.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SYSTEM_ID = 3
USER_ID = 4
ASSISTANT_ID = 5

ROLE_TO_TOKEN = {
    "system": SYSTEM_TOKEN,
    "user": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
}

ROLE_TO_ID = {
    "system": SYSTEM_ID,
    "user": USER_ID,
    "assistant": ASSISTANT_ID,
}

DEFAULT_SYSTEM = "You are a helpful assistant."


def render_message(role: Role, content: str) -> str:
    """Wrap a single message in its role token and terminator."""
    if role not in ROLE_TO_TOKEN:
        raise ValueError(f"unknown role {role!r}")
    return f"{ROLE_TO_TOKEN[role]}{content.strip()}{EOS_TOKEN}"


def build_prompt(
    messages: Sequence[Message],
    *,
    system: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Assemble a conversation into a single prompt string.

    Args:
        messages: the conversation, oldest first. Roles must be user/assistant.
        system:   system prompt. If None, DEFAULT_SYSTEM is used; pass an empty
                  string to emit an empty system turn.
        add_generation_prompt:
                  append a bare <|assistant|> so the model continues from there.
                  Use this at inference time; leave False when the final
                  assistant turn is already present (i.e. when training).

    Returns:
        A string ready to be fed to the tokenizer.
    """
    if system is None:
        system = DEFAULT_SYSTEM

    parts: List[str] = [BOS_TOKEN, render_message("system", system)]

    for msg in messages:
        role = msg["role"]
        if role == "system":
            # A system message inside the list would create a second system
            # turn. That is almost never intended, so reject it loudly.
            raise ValueError(
                "do not put a 'system' message in `messages`; use the `system=` "
                "argument instead"
            )
        parts.append(render_message(role, msg["content"]))

    if add_generation_prompt:
        parts.append(ASSISTANT_TOKEN)

    return "".join(parts)


def build_default_system_variants() -> List[str]:
    """
    A small, fixed set of system prompts.

    Keeping this list SHORT is deliberate. A 30M-parameter model cannot learn
    to follow arbitrary system prompts; giving it 3-5 consistent variants that
    it sees thousands of times each is far more likely to actually work.
    """
    return [
        "You are a helpful assistant.",
        "You are a concise assistant. Answer in one or two short sentences.",
        "You are a friendly assistant. Be warm and brief.",
        "You are a technical assistant. Explain things simply and accurately.",
    ]


def parse_conversation(text: str) -> List[Message]:
    """
    Inverse of build_prompt(), for sanity-checking data.
    Not used in the hot path -- exists so round-trip tests are easy to write.
    """
    body = text
    if body.startswith(BOS_TOKEN):
        body = body[len(BOS_TOKEN):]
    if not body.endswith(EOS_TOKEN):
        raise ValueError("conversation does not end with EOS_TOKEN")

    messages: List[Message] = []
    for chunk in body.split(EOS_TOKEN):
        if not chunk:
            continue
        for role, token in ROLE_TO_TOKEN.items():
            if chunk.startswith(token):
                messages.append(
                    {"role": role, "content": chunk[len(token):]}  # type: ignore[typeddict-item]
                )
                break
        else:
            raise ValueError(f"chunk does not start with a role token: {chunk[:40]!r}")
    return messages


def iter_assistant_spans(messages: Iterable[Message]) -> Iterable[str]:
    """Yield only the assistant contents. Used to build the SFT loss mask."""
    for msg in messages:
        if msg["role"] == "assistant":
            yield msg["content"]
