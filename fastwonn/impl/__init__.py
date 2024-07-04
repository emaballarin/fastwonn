#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# PyTorch implementation
from collections.abc import Callable
from typing import Dict
from typing import List
from warnings import warn

from torch import Tensor

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

# FAISS implementation (CPU)
try:
    from .faisscpu import _cdist_topk_faisscpu
    from .faisscpu import test_faiss_cpu

    test_faiss_cpu()
    del test_faiss_cpu
except (ImportError, AttributeError):
    warn(extra_fallback_msg("FAISS (cpu)", "torch"))
    from .fallback import _cdist_topk_faisscpu
# ──────────────────────────────────────────────────────────────────────────────

# FAISS implementation (GPU)
try:
    from .faissgpu import _cdist_topk_faissgpu
    from .faissgpu import test_faiss_gpu

    test_faiss_gpu()
    del test_faiss_gpu
except (ImportError, AttributeError):
    warn(extra_fallback_msg("FAISS (gpu)", "torch"))
    from .fallback import _cdist_topk_faissgpu
# ──────────────────────────────────────────────────────────────────────────────

call_to_impl_cdist_topk: Dict[str, Callable[[Tensor, int, bool], Tensor]] = {
    "torch": _cdist_topk_torch,
    "faissgpu": _cdist_topk_faissgpu,
    "faisscpu": _cdist_topk_faisscpu,
    "keops": _cdist_topk_keops,
}

# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = [
    "_cdist_topk_torch",
    "_cdist_topk_keops",
    "_cdist_topk_faisscpu",
    "_cdist_topk_faissgpu",
    "call_to_impl_cdist_topk",
]
