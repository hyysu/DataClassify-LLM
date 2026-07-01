from pathlib import Path
import argparse
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_classification_tool.config import DEFAULT_FIELD_TRAINING_JSON, REPORT_DIR
from data_classification_tool.evaluator import evaluate_classifier
from data_classification_tool.standalone_evaluation import evaluate_broad_label_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate field-level Bayes classifier.")
    parser.add_argument("--input", default=str(DEFAULT_FIELD_TRAINING_JSON), help="Labeled JSON/CSV path.")
    parser.add_argument("--catalog", default="", help="Optional label catalog path.")
    parser.add_argument("--output-dir", default=str(REPORT_DIR / "field_bayes_evaluation"), help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.catalog:
        result = evaluate_classifier(
            training_csv=args.input,
            catalog_csv=args.catalog,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    else:
        result = evaluate_broad_label_classifier(
            training_path=args.input,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

