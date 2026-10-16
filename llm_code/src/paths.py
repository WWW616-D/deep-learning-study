"""
Path resolution.

Every user-facing path in this project is written relative to the PROJECT ROOT,
not to the current working directory.

Why this matters
----------------
The project gets run from several places, and the working directory differs
between them:

    run.cmd                        -> cd's to the project root first
    PyCharm "Script path" runs     -> working directory is whatever the run
                                      configuration says, often the script's
                                      own folder
    plain `python src/generate.py` -> wherever the shell happened to be

Without this helper, `--tokenizer data/tokenizer/tiny_tokenizer.json` resolves
against the working directory, so running from `src/` fails with:

    tokenizer not found: data\\tokenizer\\tiny_tokenizer.json

...which reads like a missing file but is really a path-resolution bug. Anchoring
everything to ROOT makes the three invocation styles above behave identically.

Absolute paths are passed through untouched, so `--ckpt D:\\elsewhere\\x.pt`
still works.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str | Path) -> Path:
    """Return an absolute path; relative inputs are taken as project-root-relative."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def describe_missing(p: str | Path) -> str:
    """A helpful message for a path that could not be found."""
    rp = resolve(p)
    return (
        f"{rp}\n"
        f"    (resolved against the project root {ROOT};\n"
        f"     current working directory is {Path.cwd()})"
    )
