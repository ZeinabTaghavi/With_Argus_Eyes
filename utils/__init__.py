"""
Compatibility shim for older saved artifacts.

Some previously saved `joblib` / `pickle` artefacts in this repo referenced
modules under the old top-level namespace `utils.*`.

The codebase has since been reorganized and the canonical location is now:
    `with_argus_eyes.utils.*`  (under `src/with_argus_eyes/utils/`)

Unpickling requires the original import paths to be importable. This shim
aliases `utils` to `with_argus_eyes.utils` so legacy artefacts can still load.
"""

from __future__ import annotations

import sys
from importlib import import_module


def _alias_module(old: str, new: str) -> None:
    """Alias module name `old` to already-importable module `new`."""
    mod = import_module(new)
    sys.modules.setdefault(old, mod)


# Alias the package itself and a few common subpackages used in artefacts.
_alias_module("utils", "with_argus_eyes.utils")
for _sub in ("embeddings", "models", "risk", "plots"):
    _alias_module(f"utils.{_sub}", f"with_argus_eyes.utils.{_sub}")

# Make `utils` act like a package for submodule resolution.
_base = sys.modules["utils"]
if hasattr(_base, "__path__"):
    __path__ = _base.__path__  # type: ignore

