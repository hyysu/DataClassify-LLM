from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.synthetic_data import generate_synthetic_training_records


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "synthetic_data_field_bayes.joblib"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Bayes classifier from generated synthetic field metadata.")
    parser.add_argument("--records-per-category", type=int, default=60, help="Synthetic records per category.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation.")
    parser.add_argument("--samples-per-record", type=int, default=3, help="Synthetic sample values per field.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Output model path.")
    parser.add_argument(
        "--export-json",
        default="",
        help="Optional path to export generated records. Default keeps synthetic data in memory only.",
    )
    args = parser.parse_args()

    records = generate_synthetic_training_records(
        records_per_category=args.records_per_category,
        seed=args.seed,
        sample_values_per_record=args.samples_per_record,
    )
    labels = [str(record["label"]) for record in records]
    classifier = DataFieldBayesClassifier().fit(records)
    classifier.save_model(args.model)

    if args.export_json:
        output_path = Path(args.export_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"synthetic_records={len(records)}")
    print(f"label_count={len(set(labels))}")
    print(f"model={args.model}")
    if args.export_json:
        print(f"export_json={args.export_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

