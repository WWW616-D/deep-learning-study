"""
sitecustomize.py -- sandbox compatibility shim (auto-imported at startup).

Problem
-------
tempfile.mkdtemp() creates its directory with mode 0o700::

    os.mkdir(path, 0o700)

Under this DSH sandbox a directory created with 0o700 is NOT writable by the
file-access layer, so the very first write into it fails::

    PermissionError: [Errno 13] Permission denied: '...\\tmpXXXXXXXX\\some.whl'

Creating the same directory with the DEFAULT permissions works fine. Verified
directly:

    os.mkdir(path, 0o700)  -> write FAILS
    os.mkdir(path)         -> write OK
    os.mkdir(path, 0o777)  -> write OK

This breaks everything that uses tempfile.TemporaryDirectory() -- pip,
ensurepip, huggingface_hub, torch.hub -- which is most of the toolchain.

Fix
---
Reimplement mkdtemp with the mode argument removed. Nothing else changes. This
is deliberately as small as possible: one call site, one difference.

Note that mkstemp (used by NamedTemporaryFile) creates FILES with 0o600 and is
unaffected -- only directory mode 0o700 is the problem.
"""

from __future__ import annotations

import os
import tempfile

try:
    _sanitize = tempfile._sanitize_params  # type: ignore[attr-defined]
    _names = tempfile._get_candidate_names  # type: ignore[attr-defined]
    _TMP_MAX = tempfile.TMP_MAX
except AttributeError:  # pragma: no cover - defensive, CPython always has these
    _sanitize = None


def _mkdtemp_relaxed(suffix=None, prefix=None, dir=None):  # noqa: A002
    """tempfile.mkdtemp, but without the 0o700 directory mode."""
    if _sanitize is None:  # pragma: no cover
        return _ORIGINAL_MKDTEMP(suffix, prefix, dir)

    if dir is None and suffix is None and prefix is None:
        prefix, suffix, dir, output_type = _sanitize(prefix, suffix, dir)
    else:
        prefix, suffix, dir, output_type = _sanitize(prefix, suffix, dir)

    for _ in range(_TMP_MAX):
        name = next(_names())
        path = os.path.join(dir, prefix + name + suffix)
        try:
            # *** the only change: default permissions instead of 0o700 ***
            os.mkdir(path)
            return output_type(path)
        except FileExistsError:
            continue
    raise FileExistsError("no usable temporary directory name found")


_ORIGINAL_MKDTEMP = tempfile.mkdtemp

# TemporaryDirectory() resolves mkdtemp through the module global, so replacing
# the module attribute is enough to cover both APIs.
tempfile.mkdtemp = _mkdtemp_relaxed
tempfile._mkdtemp = _mkdtemp_relaxed  # type: ignore[attr-defined]
