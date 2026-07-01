"""Input and output helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data_classification_tool.models import AnalysisResult, FieldRecord


FIELD_COLUMNS = [
    "table_name",
    "column_name",
]


def _load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("data", []))
        return pd.DataFrame(payload).fillna("")
    return pd.read_csv(path).fillna("")


def _to_bool_or_none(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes", "y", "是"}:
        return True
    if text in {"false", "0", "no", "n", "否"}:
        return False
    return None


def _parse_sample_values(value: object, fallback: object = "") -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item) != "")
    if isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item) != "")

    text = str(value or "").strip()
    if text:
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return tuple(str(item) for item in parsed if str(item) != "")
            except json.JSONDecodeError:
                pass
        separators = ["||", "|", ";", "；"]
        for separator in separators:
            if separator in text:
                return tuple(item.strip() for item in text.split(separator) if item.strip())
        return (text,)

    fallback_text = str(fallback or "").strip()
    return (fallback_text,) if fallback_text else ()


def read_field_records(path: str | Path, require_label: bool = False) -> tuple[list[FieldRecord], list[str]]:
    """Read field metadata from CSV or JSON."""

    frame = _load_frame(path)
    required_columns = set(FIELD_COLUMNS)
    if require_label:
        if "label" not in frame.columns and "label_id" not in frame.columns:
            required_columns.add("label")
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    records: list[FieldRecord] = []
    labels: list[str] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            FieldRecord(
                table_name=str(row.get("table_name", "")),
                table_comment=str(row.get("table_comment", "")),
                column_name=str(row.get("column_name", "")),
                column_comment=str(row.get("column_comment", "")),
                data_type=str(row.get("data_type", "")),
                sample_value=str(row.get("sample_value", "")),
                sample_values=_parse_sample_values(row.get("sample_values", ""), row.get("sample_value", "")),
                business_context=str(row.get("business_context", "")),
                nullable=_to_bool_or_none(row.get("nullable", "")),
                is_primary_key=_to_bool_or_none(row.get("is_primary_key", "")),
                is_unique=_to_bool_or_none(row.get("is_unique", "")),
            )
        )
        if require_label:
            labels.append(str(row.get("label", row.get("label_id", ""))).strip())
    return records, labels


def flatten_result(result: AnalysisResult) -> dict[str, object]:
    """Flatten nested analysis output for CSV/Excel-friendly reporting."""

    candidates = [
        f"{candidate.label_id}:{candidate.probability:.4f}"
        for candidate in result.classification.candidates
    ]
    return {
        "table_name": result.field.table_name,
        "table_comment": result.field.table_comment,
        "column_name": result.field.column_name,
        "column_comment": result.field.column_comment,
        "data_type": result.field.data_type,
        "sample_value": result.field.sample_value,
        "business_context": result.field.business_context,
        "label_id": result.classification.label_id,
        "category": result.classification.category,
        "classification_confidence": result.classification.confidence,
        "top_candidates": " | ".join(candidates),
        "matched_patterns": " | ".join(result.classification.matched_patterns),
        "need_review": result.classification.need_review,
        "evidence_features": " | ".join(result.classification.evidence_features),
        "sample_profile": json.dumps(result.classification.sample_profile, ensure_ascii=False),
        "level": result.grading.level,
        "level_name": result.grading.level_name,
        "grading_confidence": result.grading.confidence,
        "requires_review": result.grading.requires_review,
        "controls": " | ".join(result.grading.controls),
        "reason": result.grading.reason,
        "grading_source": result.grading.source,
    }


def write_results_csv(results: list[AnalysisResult], path: str | Path) -> None:
    """Write analysis results to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([flatten_result(result) for result in results]).to_csv(path, index=False, encoding="utf-8-sig")


def write_results_json(results: list[AnalysisResult], path: str | Path) -> None:
    """Write analysis results to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
