#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List

from .pytorch import _cdist_topk_torch

# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = [
    "_cdist_topk_faisscpu",
    "_cdist_topk_faissgpu",
    "_cdist_topk_keops",
]

# ──────────────────────────────────────────────────────────────────────────────

_cdist_topk_keops = _cdist_topk_torch
_cdist_topk_faisscpu = _cdist_topk_torch
_cdist_topk_faissgpu = _cdist_topk_torch
