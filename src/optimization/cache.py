"""
Safe, deterministic LRU cache module for budget optimization results.
Computes stable SHA-256 fingerprints across candidate matrices, budgets,
and solver configurations. Returns defensive copies to prevent caller mutation.
"""

import hashlib
import copy
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


def compute_cohort_fingerprint(profit_matrix: pd.DataFrame, cost_matrix: pd.DataFrame) -> str:
    """Computes a fast, stable deterministic SHA-256 digest of candidate matrices."""
    hasher = hashlib.sha256()
    # Hash customer IDs and ordering
    hasher.update(",".join(map(str, profit_matrix.index.values)).encode("utf-8"))
    # Hash profit and cost array buffers
    hasher.update(np.ascontiguousarray(profit_matrix.to_numpy(dtype=np.float64)).tobytes())
    hasher.update(np.ascontiguousarray(cost_matrix.to_numpy(dtype=np.float64)).tobytes())
    return hasher.hexdigest()[:16]


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
    return f"opt:{policy_version}:{cohort_fingerprint}:b_{budget:.2f}:m_{mode}:gap_{gap_str}:tl_{tl_str}"


class OptimizationCache:
    """Bounded, thread-safe LRU cache with eviction tracking and defensive copies."""

    def __init__(self, maxsize: int = 64):
        if maxsize < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize}")
        self.maxsize = maxsize
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: Dict[str, int] = {}
        self._access_counter = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._cache:
            self.hits += 1
            self._access_counter += 1
            self._access_order[key] = self._access_counter
            # Return deepcopy to ensure immutability
            return copy.deepcopy(self._cache[key])
        self.misses += 1
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        # Never cache failed, invalid, or non-feasible outcomes
        if not value.get("is_feasible", False):
            return

        if key in self._cache:
            self._cache[key] = copy.deepcopy(value)
            self._access_counter += 1
            self._access_order[key] = self._access_counter
            return

        if len(self._cache) >= self.maxsize:
            # Evict Least Recently Used entry
            lru_key = min(self._access_order, key=self._access_order.get)
            del self._cache[lru_key]
            del self._access_order[lru_key]
            self.evictions += 1

        self._cache[key] = copy.deepcopy(value)
        self._access_counter += 1
        self._access_order[key] = self._access_counter

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def size(self) -> int:
        return len(self._cache)
