#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# PyTorch implementation
from typing import List
from warnings import warn

from .pytorch import _cdist_topk_torch
from .utils import extra_fallback_msg

# ──────────────────────────────────────────────────────────────────────────────

# PyKeOps implementation
try:
    from .keops import _cdist_topk_keops
except ImportError:
    warn(extra_fallback_msg("PyKeOps", "torch"))
    from .fallback import _cdist_topk_keops
# ──────────────────────────────────────────────────────────────────────────────

# FAISS implementation
try:
    from .faiss import _cdist_topk_faiss
except ImportError:
    warn(extra_fallback_msg("FAISS", "torch"))
    from .fallback import _cdist_topk_faiss
# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = ["_cdist_topk_torch", "_cdist_topk_keops", "_cdist_topk_faiss"]
