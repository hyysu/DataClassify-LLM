"""Model evaluation utilities for the Bayesian field classifier."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from data_classification_tool.catalog import get_rule, load_label_catalog
from data_classification_tool.classifier import BayesianFieldClassifier
from data_classification_tool.io import read_field_records
from data_classification_tool.models import FieldRecord


def _field_to_row(record: FieldRecord) -> dict[str, str]:
    return asdict(record)


def _report_to_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "label": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1_score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )
        else:
            rows.append({"label": label, "accuracy": values})
    return pd.DataFrame(rows)


def _write_evaluation_workbook(
    output_path: Path,
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    confusion: pd.DataFrame,
    report: pd.DataFrame,
    label_counts: pd.DataFrame,
    confusing_pairs: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="summary")
        predictions.to_excel(writer, index=False, sheet_name="predictions")
        confusion.to_excel(writer, sheet_name="confusion_matrix")
        report.to_excel(writer, index=False, sheet_name="classification_report")
        label_counts.to_excel(writer, index=False, sheet_name="label_counts")
        confusing_pairs.to_excel(writer, index=False, sheet_name="confusing_pairs")


def _confusing_pairs(confusion: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expected_label in confusion.index:
        for predicted_label in confusion.columns:
            if expected_label == predicted_label:
                continue
            count = int(confusion.loc[expected_label, predicted_label])
            if count > 0:
                rows.append(
                    {
                        "expected_label": expected_label,
                        "predicted_label": predicted_label,
                        "confusion_count": count,
                    }
                )
    return pd.DataFrame(rows).sort_values("confusion_count", ascending=False) if rows else pd.DataFrame(
        columns=["expected_label", "predicted_label", "confusion_count"]
    )


def evaluate_classifier(
    training_csv: str | Path,
    catalog_csv: str | Path,
    output_dir: str | Path,
    test_size: float = 0.4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate the classifier with a stratified train/test split.

    The function writes four artifacts:
    - evaluation_metrics.json
    - evaluation_predictions.csv
    - confusion_matrix.csv
    - evaluation_report.xlsx
    """

    records, labels = read_field_records(training_csv, require_label=True)
    catalog = load_label_catalog(catalog_csv)

    train_records, test_records, train_labels, test_labels = train_test_split(
        records,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    classifier = BayesianFieldClassifier().fit(train_records, train_labels)

    prediction_rows: list[dict[str, Any]] = []
    predicted_labels: list[str] = []
    for record, expected_label in zip(test_records, test_labels):
        result = classifier.predict_one(record, catalog)
        predicted_labels.append(result.label_id)
        expected_rule = get_rule(catalog, expected_label)
        predicted_rule = get_rule(catalog, result.label_id)
        prediction_rows.append(
            {
                **_field_to_row(record),
                "expected_label": expected_label,
                "expected_category": expected_rule.category,
                "predicted_label": result.label_id,
                "predicted_category": predicted_rule.category,
                "confidence": result.confidence,
                "is_correct": expected_label == result.label_id,
                "top_candidates": " | ".join(
                    f"{candidate.label_id}:{candidate.probability:.4f}"
                    for candidate in result.candidates
                ),
                "matched_patterns": " | ".join(result.matched_patterns),
                "need_review": result.need_review,
                "evidence_features": " | ".join(result.evidence_features),
            }
        )

    label_order = sorted(set(labels))
    report_dict = classification_report(
        test_labels,
        predicted_labels,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    confusion = pd.DataFrame(
        confusion_matrix(test_labels, predicted_labels, labels=label_order),
        index=label_order,
        columns=label_order,
    )
    confusion.index.name = "expected_label"
    predictions = pd.DataFrame(prediction_rows)
    report = _report_to_frame(report_dict)
    label_counts = (
        pd.Series(labels)
        .value_counts()
        .rename_axis("label")
        .reset_index(name="sample_count")
        .sort_values("label")
    )
    confusing_pairs = _confusing_pairs(confusion)

    summary = {
        "sample_count": len(records),
        "train_count": len(train_records),
        "test_count": len(test_records),
        "label_count": len(label_order),
        "test_size": test_size,
        "random_state": random_state,
        "accuracy": round(float(accuracy_score(test_labels, predicted_labels)), 4),
        "macro_f1": round(float(f1_score(test_labels, predicted_labels, average="macro")), 4),
        "weighted_f1": round(float(f1_score(test_labels, predicted_labels, average="weighted")), 4),
        "label_sample_counts": dict(zip(label_counts["label"], label_counts["sample_count"])),
        "top_confusing_pairs": confusing_pairs.head(10).to_dict(orient="records"),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    predictions_path = output_dir / "evaluation_predictions.csv"
    confusion_path = output_dir / "confusion_matrix.csv"
    workbook_path = output_dir / "evaluation_report.xlsx"

    metrics_payload = {"summary": summary, "classification_report": report_dict}
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    confusion.to_csv(confusion_path, encoding="utf-8-sig")
    _write_evaluation_workbook(workbook_path, summary, predictions, confusion, report, label_counts, confusing_pairs)

    return {
        "summary": summary,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "confusion_path": str(confusion_path),
        "workbook_path": str(workbook_path),
    }
