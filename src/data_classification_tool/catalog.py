"""Load and query the field label catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_classification_tool.models import LabelRule


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _split_pipe(value: object) -> list[str]:
    text = str(value or "").strip()
    return [item.strip() for item in text.split("|") if item.strip()]


def load_label_catalog(path: str | Path) -> dict[str, LabelRule]:
    """Load label rules from a CSV catalog."""

    frame = pd.read_csv(path).fillna("")
    catalog: dict[str, LabelRule] = {}
    for row in frame.to_dict(orient="records"):
        rule = LabelRule(
            label_id=str(row["label_id"]).strip(),
            category=str(row["category"]).strip(),
            description=str(row["description"]).strip(),
            default_level=str(row["default_level"]).strip(),
            is_personal_info=_to_bool(row["is_personal_info"]),
            is_sensitive_personal_info=_to_bool(row["is_sensitive_personal_info"]),
            important_data_hint=_to_bool(row["important_data_hint"]),
            controls=_split_pipe(row["controls"]),
            keywords=_split_pipe(row["keywords"]),
        )
        catalog[rule.label_id] = rule
    return catalog


def get_rule(catalog: dict[str, LabelRule], label_id: str) -> LabelRule:
    """Return a label rule, falling back to the unknown rule if needed."""

    if label_id in catalog:
        return catalog[label_id]
    if "unknown" in catalog:
        return catalog["unknown"]
    raise KeyError(f"Label not found and no unknown fallback exists: {label_id}")

