"""
Persistence module for verified budget-frontier optimization results.
Stores metadata, cohort fingerprints, solver configurations, and customer allocations
with validation to prevent corrupted, mismatched, or stale artifact reuse.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

FRONTIER_SCHEMA_VERSION = "frontier_v1"


def save_frontier_result(
    output_dir: str,
    cohort_fingerprint: str,
    cohort_id: str,
    budget: float,
    mode: str,
    solver_settings: Dict[str, Any],
    response_dict: Dict[str, Any],
    policy_version: str = "v1",
) -> str:
    """Saves a verified optimization response to disk with metadata and allocations."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"frontier_{cohort_fingerprint[:8]}_b{int(budget)}_{mode}.json"
    filepath = os.path.join(output_dir, filename)

    payload = {
        "schema_version": FRONTIER_SCHEMA_VERSION,
        "policy_version": policy_version,
        "cohort_fingerprint": cohort_fingerprint,
        "cohort_id": cohort_id,
        "budget": budget,
        "mode": mode,
        "solver_settings": solver_settings,
        "saved_at_epoch": time.time(),
        "response": response_dict,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return filepath


def load_frontier_result(
    filepath: str,
    expected_fingerprint: str,
    expected_budget: float,
    expected_mode: str,
    expected_policy_version: str = "v1",
) -> Optional[Dict[str, Any]]:
    """Loads and validates a persisted frontier point. Returns None if invalid or mismatched."""
    if not os.path.isfile(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # Strict validation of schema, fingerprint, budget, and mode
    if data.get("schema_version") != FRONTIER_SCHEMA_VERSION:
        return None
    if data.get("policy_version") != expected_policy_version:
        return None
    if data.get("cohort_fingerprint") != expected_fingerprint:
        return None
    if abs(float(data.get("budget", -1)) - expected_budget) > 1e-4:
        return None
    if data.get("mode") != expected_mode:
        return None

    resp = data.get("response")
    if not resp or not resp.get("is_feasible", False):
        return None

    return resp
