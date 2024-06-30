#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["extra_fallback_msg"]


# ──────────────────────────────────────────────────────────────────────────────
def extra_fallback_msg(requested: str, fallback: str) -> str:
    return f"Extra requirement '{requested}' not found. Falling back to the {fallback} implementation."
