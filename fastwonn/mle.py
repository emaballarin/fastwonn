#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from contextlib import ExitStack

import torch
from torch import Tensor

from .impl import call_to_impl_cdist_topk

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["mle_id", "mle_id_avg"]


# ──────────────────────────────────────────────────────────────────────────────
def mle_id(
    x: Tensor,
    nneigh: int = 2,
    twonn_fix: bool = False,
    differentiable: bool = False,
    impl: str = "torch",
) -> Tensor:

    with ExitStack() as stack:
        stack.enter_context(torch.no_grad()) if not differentiable else None

        ks: Tensor = call_to_impl_cdist_topk[impl](x, nneigh, False)[:, 1:]

        if twonn_fix and nneigh == 2:
            return -2 * ks.size(0) / torch.log(torch.div(*torch.unbind(ks, 1))).sum()

        return (2 * (nneigh - 1) / torch.log(ks[:, -1].view(-1, 1) / ks).sum(1)).mean()


# ──────────────────────────────────────────────────────────────────────────────


def mle_id_avg(
    x: Tensor,
    nneigh_min: int = 2,
    nneigh_max: int = 10,
    twonn_fix: bool = False,
    differentiable: bool = False,
    impl: str = "torch",
) -> Tensor:

    with ExitStack() as stack:
        stack.enter_context(torch.no_grad()) if not differentiable else None

        twonn_sep: bool = twonn_fix and nneigh_min == 2

        ks: Tensor = call_to_impl_cdist_topk[impl](x, nneigh_max, False)[:, 1:]
        runs = [
            (
                2
                * (nneigh_max - 1 - i)
                / torch.log(
                    ks[:, -1 - i].view(-1, 1) / (ks[:, :-i] if i != 0 else ks)
                ).sum(1)
            ).mean()
            for i in range(nneigh_max - nneigh_min + (not twonn_sep))
        ]

        if twonn_sep:
            runs.append(
                -2
                * ks.size(0)
                / torch.log(torch.div(*torch.unbind(ks[:, 0:2], 1))).sum()
            )

        return torch.stack(runs).nanmean()
