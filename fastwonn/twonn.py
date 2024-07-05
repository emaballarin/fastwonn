#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from contextlib import ExitStack
from math import floor

import torch
from torch import Tensor
from torchsort import soft_sort as ssort

from .impl import call_to_impl_cdist_topk
from .utils import _refine_k1k2

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["twonn_id"]


# ──────────────────────────────────────────────────────────────────────────────
def twonn_id(
    x: Tensor,
    fraction: float = 0.9,
    x_distances: bool = False,
    mle_fit: bool = False,
    differentiable: bool = False,
    impl: str = "torch",
) -> Tensor:

    with ExitStack() as stack:
        stack.enter_context(torch.no_grad()) if not differentiable else None

        ks: Tensor = call_to_impl_cdist_topk[impl](x, 2, x_distances)
        k1, k2 = torch.unbind(ks, 1)[1:]
        k1, k2 = _refine_k1k2(k1, k2)
        lenk1: int = len(k1)
        presort: Tensor = (k2 / k1).flatten()

        if mle_fit:
            return 2 * lenk1 / torch.log(presort).sum()

        npoints: int = floor(lenk1 * fraction)
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
