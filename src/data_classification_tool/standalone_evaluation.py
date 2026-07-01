"""Evaluation for broad-label sample data without a label catalog."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.io import read_field_records


def _confusing_pairs(confusion: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expected_label in confusion.index:
        for predicted_label in confusion.columns:
            if expected_label == predicted_label:
                continue
            count = int(confusion.loc[expected_label, predicted_label])
            if count:
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


def _resolve_split(labels: list[str], test_size: float | int) -> tuple[int, list[str] | None]:
    """Return a safe test count and optional stratify labels.

    Small field-classification demos often have many labels but only a few
    examples per label. Stratified splitting requires both train and test sets
    to contain at least one example from every class.
    """

    sample_count = len(labels)
    if sample_count < 2:
        raise ValueError("at least two labeled records are required for evaluation")

    label_counts = Counter(labels)
    label_count = len(label_counts)
    if isinstance(test_size, float) and 0 < test_size < 1:
        test_count = math.ceil(sample_count * test_size)
    else:
        test_count = int(test_size)
    test_count = max(1, min(test_count, sample_count - 1))

    can_stratify = all(count >= 2 for count in label_counts.values()) and sample_count >= label_count * 2
    if not can_stratify:
        return test_count, None

    test_count = max(test_count, label_count)
    test_count = min(test_count, sample_count - label_count)
    return test_count, labels


def evaluate_broad_label_classifier(
    training_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.4,
    random_state: int = 42,
) -> dict[str, Any]:
    records, labels = read_field_records(training_path, require_label=True)
    resolved_test_size, stratify_labels = _resolve_split(labels, test_size)
    train_records, test_records, train_labels, test_labels = train_test_split(
        records,
        labels,
        test_size=resolved_test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    classifier = DataFieldBayesClassifier().fit(train_records, train_labels)
    predictions = classifier.predict_batch(test_records)
    predicted_labels = [prediction.predicted_category for prediction in predictions]
    label_order = sorted(set(labels))

    confusion = pd.DataFrame(
        confusion_matrix(test_labels, predicted_labels, labels=label_order),
        index=label_order,
        columns=label_order,
    )
    confusion.index.name = "expected_label"
    report_dict = classification_report(
        test_labels,
        predicted_labels,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    label_counts = (
        pd.Series(labels)
        .value_counts()
        .rename_axis("label")
        .reset_index(name="sample_count")
        .sort_values("label")
    )
    confusing_pairs = _confusing_pairs(confusion)
    prediction_rows = []
    for record, expected, prediction in zip(test_records, test_labels, predictions):
        prediction_rows.append(
            {
                "table_name": record.table_name,
                "table_comment": record.table_comment,
                "column_name": record.column_name,
                "column_comment": record.column_comment,
                "data_type": record.data_type,
                "expected_label": expected,
                "predicted_label": prediction.predicted_category,
                "confidence": prediction.confidence,
                "need_review": prediction.need_review,
                "top_candidates": " | ".join(
                    f"{candidate.category}:{candidate.probability:.4f}"
                    for candidate in prediction.top_candidates
                ),
                "evidence_features": " | ".join(prediction.evidence_features),
            }
        )

    summary = {
        "sample_count": len(records),
        "train_count": len(train_records),
        "test_count": len(test_records),
        "label_count": len(label_order),
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

    metrics_path.write_text(
        json.dumps({"summary": summary, "classification_report": report_dict}, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False, encoding="utf-8-sig")
    confusion.to_csv(confusion_path, encoding="utf-8-sig")
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="summary")
        pd.DataFrame(prediction_rows).to_excel(writer, index=False, sheet_name="predictions")
        confusion.to_excel(writer, sheet_name="confusion_matrix")
        label_counts.to_excel(writer, index=False, sheet_name="label_counts")
        confusing_pairs.to_excel(writer, index=False, sheet_name="confusing_pairs")

    return {
        "summary": summary,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "confusion_path": str(confusion_path),
        "workbook_path": str(workbook_path),
    }
