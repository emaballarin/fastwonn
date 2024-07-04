#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = ["_cdist_topk_torch"]

# ──────────────────────────────────────────────────────────────────────────────


@torch.jit.script
def _cdist_topk_torch(x: Tensor, k: int = 2, x_distances: bool = False) -> Tensor:
    xcd: Tensor = torch.cdist(x, x) if not x_distances else x
    return torch.topk(xcd, k + 1, 1, largest=False)[0] ** 2
