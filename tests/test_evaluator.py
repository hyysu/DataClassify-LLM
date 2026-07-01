from pathlib import Path

from data_classification_tool.evaluator import evaluate_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CSV = PROJECT_ROOT / "data" / "training_fields.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "field_label_catalog.csv"


def test_evaluate_classifier_writes_reports(tmp_path: Path) -> None:
    result = evaluate_classifier(
        training_csv=TRAINING_CSV,
        catalog_csv=CATALOG_CSV,
        output_dir=tmp_path,
        test_size=0.4,
        random_state=42,
    )

    summary = result["summary"]

    assert summary["sample_count"] > summary["test_count"]
    assert summary["accuracy"] >= 0.7
    assert Path(result["metrics_path"]).exists()
    assert Path(result["predictions_path"]).exists()
    assert Path(result["confusion_path"]).exists()
    assert Path(result["workbook_path"]).exists()

