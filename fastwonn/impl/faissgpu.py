#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List

import faiss
import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = ["_cdist_topk_faissgpu", "test_faiss_gpu"]

# ──────────────────────────────────────────────────────────────────────────────


def test_faiss_gpu() -> None:
    _ = faiss.GpuIndexFlatL2


# ──────────────────────────────────────────────────────────────────────────────


# noinspection PyArgumentList
def _cdist_topk_faissgpu(x: Tensor, k: int = 2, x_distances: bool = False) -> Tensor:
    if x_distances:
        raise NotImplementedError(
            "FAISS implementation does not support distance matrices. Use the 'torch' implementation instead."
        )
    xdev: torch.device = x.device
    x: Tensor = x.cpu()
    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), x.shape[1])
    index.add(x)
    return torch.from_numpy(index.search(x, k + 1)[0]).to(xdev)
