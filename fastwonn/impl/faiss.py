#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List

import faiss
import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = ["_cdist_topk_faiss"]

# ──────────────────────────────────────────────────────────────────────────────


# noinspection PyArgumentList
def _cdist_topk_faiss(
    x: Tensor, x_distances: bool = False, cuda: bool = True
) -> Tensor:
    if x_distances:
        raise NotImplementedError(
            "FAISS implementation does not support distance matrices. Use the 'torch' implementation instead."
        )
    xdev: torch.device = x.device
    x: Tensor = x.cpu()
    xts: int = x.shape[1]
    index = (
        faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), xts)
        if cuda
        else faiss.IndexFlatL2(xts)
    )
    index.add(x)
    return torch.from_numpy(index.search(x, 3)[0]).to(xdev)
