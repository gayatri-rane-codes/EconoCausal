"""
Deterministic in-memory LRU cache for optimization results.
Prevents cross-cohort, cross-budget, and cross-mode key collisions via SHA-256 fingerprinting.
"""

import copy
import hashlib
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def compute_cohort_fingerprint(profit_df: pd.DataFrame, cost_df: pd.DataFrame) -> str:
    """
    Computes a deterministic SHA-256 hash over candidate customer matrices.
    Rounds numerical matrices to 6 decimal places to prevent bit-level floating-point jitter
    from multi-threaded estimation models while preserving strict numerical discrimination.
    """
    hasher = hashlib.sha256()
    hasher.update(str(list(profit_df.index)).encode("utf-8"))
    hasher.update(str(list(profit_df.columns)).encode("utf-8"))
    hasher.update(np.round(profit_df.to_numpy(dtype=np.float64), 6).tobytes())
    hasher.update(np.round(cost_df.to_numpy(dtype=np.float64), 6).tobytes())
    return hasher.hexdigest()


def build_cache_key(
    cohort_fingerprint: str,
    budget: float,
    mode: str,
    mip_rel_gap: Optional[float] = None,
    time_limit: Optional[float] = None,
    policy_version: str = "v1",
) -> str:
    """Generates a unique deterministic string key for all result-affecting parameters."""
    gap_str = f"{mip_rel_gap:.6f}" if mip_rel_gap is not None else "none"
    tl_str = f"{time_limit:.2f}" if time_limit is not None else "none"
    return f"{cohort_fingerprint}_b{budget:.4f}_m{mode}_g{gap_str}_t{tl_str}_p{policy_version}"


class OptimizationCache:
    """Thread-safe, bounded LRU cache for optimization results returning defensive copies."""

    def __init__(self, maxsize: int = 64):
        if maxsize < 1:
            raise ValueError(f"Cache maxsize must be at least 1, got {maxsize}")
        self.maxsize = maxsize
        self._cache: Dict[str, Any] = {}
        self._access_order: Dict[str, int] = {}
        self._access_counter = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            self.misses += 1
            return None
        self.hits += 1
        self._access_counter += 1
        self._access_order[key] = self._access_counter
        return copy.deepcopy(self._cache[key])

    def put(self, key: str, value: Any) -> None:
        if hasattr(value, "is_feasible"):
            is_feas = getattr(value, "is_feasible")
        elif isinstance(value, dict):
            is_feas = value.get("is_feasible", False)
        else:
            is_feas = False

        if not is_feas:
            return

        if key in self._cache:
            self._cache[key] = copy.deepcopy(value)
            self._access_counter += 1
            self._access_order[key] = self._access_counter
            return

        if len(self._cache) >= self.maxsize:
            lru_key = min(self._access_order, key=self._access_order.get)
            del self._cache[lru_key]
            del self._access_order[lru_key]
            self.evictions += 1

        self._cache[key] = copy.deepcopy(value)
        self._access_counter += 1
        self._access_order[key] = self._access_counter

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
        self._access_counter = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._cache)
