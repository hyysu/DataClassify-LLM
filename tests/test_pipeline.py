from pathlib import Path

from data_classification_tool.pipeline import analyze_fields, train_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CSV = PROJECT_ROOT / "data" / "training_fields.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "field_label_catalog.csv"
DEMO_CSV = PROJECT_ROOT / "data" / "demo_fields.csv"


def test_train_and_analyze_demo(tmp_path: Path) -> None:
    model_path = tmp_path / "field_classifier.joblib"
    output_csv = tmp_path / "report.csv"
    output_json = tmp_path / "report.json"

    train_classifier(TRAINING_CSV, model_path)
    results = analyze_fields(
        input_csv=DEMO_CSV,
        catalog_csv=CATALOG_CSV,
        model_path=model_path,
        output_csv=output_csv,
        output_json=output_json,
        grader_name="rule",
    )

    assert model_path.exists()
    assert output_csv.exists()
    assert output_json.exists()
    assert len(results) == 8

    by_column = {result.field.column_name: result for result in results}
    assert by_column["phone_number"].classification.label_id == "mobile_phone"
    assert by_column["client_secret"].grading.level == "L4"
    assert by_column["public_title"].grading.level == "L1"

