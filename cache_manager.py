"""
cache_manager.py
────────────────
Handles persistent on-disk caching of all computed pipeline results.
On first run  → fetch from Supabase + compute → save to disk
On later runs → load from disk, skip all DB/compute work entirely.
"""

import os
import json
import pickle
import hashlib
import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np

CACHE_DIR = Path(__file__).parent / ".pipeline_cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_MANIFEST = CACHE_DIR / "manifest.json"


def _manifest() -> dict:
    if CACHE_MANIFEST.exists():
        with open(CACHE_MANIFEST) as f:
            return json.load(f)
    return {}


def _save_manifest(m: dict):
    with open(CACHE_MANIFEST, "w") as f:
        json.dump(m, f, indent=2, default=str)


def cache_exists(key: str) -> bool:
    m = _manifest()
    if key not in m:
        return False
    path = CACHE_DIR / m[key]["file"]
    return path.exists()


def load_cache(key: str) -> Any:
    m = _manifest()
    if key not in m:
        return None
    path = CACHE_DIR / m[key]["file"]
    if not path.exists():
        return None
    ext = path.suffix
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ext == ".json":
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(key: str, data: Any, label: str = ""):
    # Pick storage format by type
    if isinstance(data, pd.DataFrame):
        fname = f"{key}.parquet"
        path  = CACHE_DIR / fname
        data.to_parquet(path, index=True)
    elif isinstance(data, dict):
        fname = f"{key}.json"
        path  = CACHE_DIR / fname
        # Serialise any non-JSON-safe values
        safe = _make_json_safe(data)
        with open(path, "w") as f:
            json.dump(safe, f, indent=2, default=str)
    else:
        fname = f"{key}.pkl"
        path  = CACHE_DIR / fname
        with open(path, "wb") as f:
            pickle.dump(data, f)

    m = _manifest()
    m[key] = {
        "file":    fname,
        "label":   label,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    _save_manifest(m)


def clear_cache():
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(exist_ok=True)


def cache_info() -> dict:
    """Return metadata about every cached artefact."""
    m = _manifest()
    info = {}
    for key, meta in m.items():
        path = CACHE_DIR / meta["file"]
        size_kb = path.stat().st_size / 1024 if path.exists() else 0
        info[key] = {**meta, "size_kb": round(size_kb, 1)}
    return info


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_json_safe(obj):
    """Recursively convert numpy/pandas types to plain Python."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    return obj
