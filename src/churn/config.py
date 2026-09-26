"""Configuration loading and calendar helpers."""

from __future__ import annotations

import calendar
import copy
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the YAML config and apply nested overrides such as {"simulation": {"n_subscribers": 1000}}."""
    with open(path or DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_month(month: str | date) -> date:
    """'2025-10' -> date(2025, 10, 1)."""
    if isinstance(month, date):
        return month.replace(day=1)
    year, mon = month.split("-")[:2]
    return date(int(year), int(mon), 1)


def add_months(month: date, n: int) -> date:
    idx = month.year * 12 + (month.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def month_end(month: date) -> date:
    return month.replace(day=calendar.monthrange(month.year, month.month)[1])


def month_label(month: date) -> str:
    return f"{month.year:04d}-{month.month:02d}"
