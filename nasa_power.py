"""
GreenScore — NASA POWER API Integration
=========================================
Fetches monthly climate data (temperature, precipitation) from the
NASA POWER REST API for any GPS coordinate, then engineers physical
risk features: flood frequency, drought severity, temperature anomaly,
and extreme weather event counts.

API Reference: https://power.larc.nasa.gov/docs/

Includes a JSON file cache so that repeated runs or dashboard reloads
do not re-call the API for locations already fetched.
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────
_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nasa_cache.json')
_memory_cache: Dict[str, dict] = {}


def _load_disk_cache() -> Dict[str, dict]:
    """Load the on-disk JSON cache into memory (once)."""
    global _memory_cache
    if _memory_cache:
        return _memory_cache
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, 'r') as f:
                _memory_cache = json.load(f)
            logger.info("Loaded NASA cache with %d entries.", len(_memory_cache))
        except (json.JSONDecodeError, IOError):
            _memory_cache = {}
    return _memory_cache


def _save_disk_cache() -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, 'w') as f:
        json.dump(_memory_cache, f, indent=2)


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 1)}_{round(lon, 1)}"


# ─────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────
_API_URL = 'https://power.larc.nasa.gov/api/temporal/monthly/point'


def fetch_climate_data(
    lat: float,
    lon: float,
    start_year: int = 2010,
    end_year: int = 2023,
) -> Optional[dict]:
    """
    Fetch monthly T2M and PRECTOTCORR from NASA POWER for a point.

    Returns the ``properties.parameter`` dict on success, or ``None``
    on failure.  Results are cached to ``data/nasa_cache.json``.
    """
    cache = _load_disk_cache()
    key = _cache_key(lat, lon)
    if key in cache:
        return cache[key].get('_raw')

    try:
        resp = requests.get(
            _API_URL,
            params={
                'parameters': 'T2M,PRECTOTCORR',
                'community': 'RE',
                'longitude': round(lon, 1),
                'latitude': round(lat, 1),
                'start': start_year,
                'end': end_year,
                'format': 'JSON',
            },
            timeout=30,
        )
        if resp.status_code == 200:
            raw = resp.json()['properties']['parameter']
            # store raw + derived features together
            features = engineer_physical_features(raw)
            features['_raw'] = raw
            cache[key] = features
            _save_disk_cache()
            return raw
    except Exception as exc:
        logger.warning("NASA API error (%s, %s): %s", lat, lon, exc)
    return None


# ─────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────
_FALLBACK_FEATURES = {
    'flood_freq_score': 0.30,
    'drought_severity_index': 0.30,
    'temp_anomaly_5yr': 0.0,
    'extreme_weather_events_count': 2,
    'physical_risk_score': 0.30,
}


def engineer_physical_features(raw: Optional[dict]) -> dict:
    """
    Convert raw NASA monthly climate arrays into five physical-risk
    features used by the XGBoost model.

    Features
    --------
    flood_freq_score            : fraction of months with precipitation > 95th pctl
    drought_severity_index      : normalised deviation of min precip from mean
    temp_anomaly_5yr            : mean(last 5 yr) − mean(earlier years) in °C
    extreme_weather_events_count: # months with temp OR precip above 95th pctl
    physical_risk_score         : weighted composite (0–1)
    """
    if not raw:
        return dict(_FALLBACK_FEATURES)

    temp = [v for v in raw.get('T2M', {}).values() if v not in (-999, -999.0)]
    precip = [v for v in raw.get('PRECTOTCORR', {}).values() if v not in (-999, -999.0)]

    if len(temp) < 12 or len(precip) < 12:
        return dict(_FALLBACK_FEATURES)

    # Flood frequency — months above 95th-pctl precipitation
    p95 = np.percentile(precip, 95)
    flood_freq = sum(1 for p in precip if p > p95) / len(precip)

    # Drought severity index
    mean_p = np.mean(precip)
    std_p = np.std(precip)
    drought = max(0.0, (mean_p - np.min(precip)) / (mean_p + std_p + 1e-6))
    drought = min(drought, 1.0)

    # Temperature anomaly — recent 5 years vs earlier
    split = min(60, len(temp) - 12)  # 60 months = 5 years
    if split > 0:
        temp_anom = float(np.mean(temp[-split:]) - np.mean(temp[:-split]))
    else:
        temp_anom = 0.0

    # Extreme weather events (heat + flood extremes)
    t95 = np.percentile(temp, 95)
    extreme_events = sum(1 for t in temp if t > t95) + sum(1 for p in precip if p > p95)

    # Composite physical risk score (0–1)
    phys = (
        0.35 * min(flood_freq, 1.0)
        + 0.30 * drought
        + 0.20 * min(max(temp_anom / 3.0, 0), 1.0)
        + 0.15 * min(extreme_events / 20.0, 1.0)
    )

    return {
        'flood_freq_score': round(flood_freq, 4),
        'drought_severity_index': round(drought, 4),
        'temp_anomaly_5yr': round(temp_anom, 4),
        'extreme_weather_events_count': int(extreme_events),
        'physical_risk_score': round(min(phys, 1.0), 4),
    }


# ─────────────────────────────────────────────────────────
# Batch helper — enrich a DataFrame
# ─────────────────────────────────────────────────────────

def enrich_with_climate_features(
    locations: list[Tuple[float, float]],
    delay: float = 0.4,
) -> Dict[str, dict]:
    """
    Fetch + engineer climate features for a list of (lat, lon) tuples.

    Returns a dict keyed by ``"lat_lon"`` with feature dicts as values.
    Cached locations are not re-fetched.
    """
    cache = _load_disk_cache()
    results: Dict[str, dict] = {}
    to_fetch = []

    for lat, lon in locations:
        key = _cache_key(lat, lon)
        if key in cache:
            results[key] = {k: v for k, v in cache[key].items() if k != '_raw'}
        else:
            to_fetch.append((lat, lon))

    if to_fetch:
        logger.info("Fetching climate data for %d new locations from NASA POWER…", len(to_fetch))
        for i, (lat, lon) in enumerate(to_fetch):
            key = _cache_key(lat, lon)
            raw = fetch_climate_data(lat, lon)
            features = cache.get(key, engineer_physical_features(raw))
            results[key] = {k: v for k, v in features.items() if k != '_raw'}
            if i < len(to_fetch) - 1:
                time.sleep(delay)
        logger.info("NASA POWER fetch complete.")
    else:
        logger.info("All %d locations served from cache.", len(locations))

    return results


def get_physical_features_for_state(
    state: str,
    coord_map_us: dict,
    coord_map_india: dict,
) -> dict:
    """
    Look up (or fetch) physical-risk features for a state name/code.

    Falls back to ``_FALLBACK_FEATURES`` if the state is unknown or
    the API call fails.
    """
    coords = coord_map_us.get(state) or coord_map_india.get(state)
    if coords is None:
        return dict(_FALLBACK_FEATURES)

    lat, lon = coords[0], coords[1]
    cache = _load_disk_cache()
    key = _cache_key(lat, lon)

    if key in cache:
        return {k: v for k, v in cache[key].items() if k != '_raw'}

    raw = fetch_climate_data(lat, lon)
    features = cache.get(key, engineer_physical_features(raw))
    return {k: v for k, v in features.items() if k != '_raw'}
