from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _PACKAGE_ROOT.parent / "src" / "llmgenecircuitdiscovery"

if str(_SRC_PACKAGE.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_PACKAGE.parent))

__path__ = [str(_SRC_PACKAGE)]

from llmgenecircuitdiscovery.cli import main

ad.settings.allow_write_nullable_strings = True

__all__ = ["main"]
