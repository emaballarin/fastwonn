#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial as fpartial
from math import floor
from typing import Dict
from typing import Tuple

import torch
from torch import Tensor
from torchsort import soft_sort as ssort

from .impl import _cdist_topk_faiss
from .impl import _cdist_topk_keops
from .impl import _cdist_topk_torch

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["twonn_id"]
# ──────────────────────────────────────────────────────────────────────────────
_call_to_impl: Dict[str, Callable[[Tensor, bool], Tensor]] = {
    "torch": _cdist_topk_torch,
    "faissgpu": fpartial(_cdist_topk_faiss, cuda=True),
    "faisscpu": fpartial(_cdist_topk_faiss, cuda=False),
    "keops": _cdist_topk_keops,
}


# ──────────────────────────────────────────────────────────────────────────────
# noinspection PyUnresolvedReferences
@torch.jit.script
def _refine_k1k2(k1: Tensor, k2: Tensor) -> Tuple[Tensor, Tensor]:
    idxs: Tensor = (k1 != 0).logical_and(k1 != k2)
    return k1[idxs], k2[idxs]


# ──────────────────────────────────────────────────────────────────────────────
def twonn_id(
    x: Tensor,
    fraction: float = 0.9,
    x_distances: bool = False,
    differentiable: bool = False,
    impl: str = "torch",
) -> Tensor:

    with ExitStack() as stack:
        stack.enter_context(torch.no_grad()) if not differentiable else None

        ks: Tensor = _call_to_impl[impl](x, x_distances)
        k1, k2 = torch.unbind(ks, 1)[1:]
        k1, k2 = _refine_k1k2(k1, k2)
        lenk1: int = len(k1)
        npoints: int = floor(lenk1 * fraction)
        presort: Tensor = torch.divide(k2, k1).flatten()
        mu: Tensor = (
            ssort(presort.view(1, -1)) if differentiable else torch.sort(presort)
        )[0][:npoints]
        femp: Tensor = torch.arange(
            1 / lenk1, 1 + 1 / lenk1, 1 / lenk1, dtype=x.dtype, device=x.device
        )[:npoints]
        logmu: Tensor = torch.log(mu)
        logfemp: Tensor = torch.log(1 - femp)
        slope = torch.linalg.lstsq(logmu.unsqueeze(-1), -logfemp.unsqueeze(-1))
        return slope.solution.squeeze() * 2
