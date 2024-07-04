#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import Tuple

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────

__all__ = ["_refine_k1k2"]

# ──────────────────────────────────────────────────────────────────────────────


# noinspection PyUnresolvedReferences
@torch.jit.script
def _refine_k1k2(k1: Tensor, k2: Tensor) -> Tuple[Tensor, Tensor]:
    idxs: Tensor = (k1 != 0).logical_and(k1 != k2)
    return k1[idxs], k2[idxs]
