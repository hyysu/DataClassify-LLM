"""Command line interface for the data classification prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_classification_tool.evaluator import evaluate_classifier
from data_classification_tool.pipeline import analyze_fields, train_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING = PROJECT_ROOT / "data" / "training_fields.csv"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "field_label_catalog.csv"
DEFAULT_DEMO = PROJECT_ROOT / "data" / "demo_fields.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "field_classifier.joblib"
DEFAULT_REPORT_CSV = PROJECT_ROOT / "reports" / "demo_report.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "demo_report.json"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "reports" / "evaluation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Field-level data classification and grading tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Bayesian field classifier.")
    train_parser.add_argument("--input", default=str(DEFAULT_TRAINING), help="Training CSV path.")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model output path.")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze field metadata from CSV.")
    analyze_parser.add_argument("--input", default=str(DEFAULT_DEMO), help="Input field metadata CSV path.")
    analyze_parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="Label catalog CSV path.")
    analyze_parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained model path.")
    analyze_parser.add_argument("--output-csv", default=str(DEFAULT_REPORT_CSV), help="CSV report path.")
    analyze_parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON), help="JSON report path.")
    analyze_parser.add_argument("--grader", choices=["rule", "llm"], default="rule", help="Grading backend.")
    analyze_parser.add_argument("--auto-train", action="store_true", help="Train the model first if it does not exist.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate classifier accuracy on labeled samples.")
    evaluate_parser.add_argument("--input", default=str(DEFAULT_TRAINING), help="Labeled training CSV path.")
    evaluate_parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="Label catalog CSV path.")
    evaluate_parser.add_argument("--output-dir", default=str(DEFAULT_EVALUATION_DIR), help="Evaluation output directory.")
    evaluate_parser.add_argument("--test-size", type=float, default=0.4, help="Test split ratio.")
    evaluate_parser.add_argument("--random-state", type=int, default=42, help="Random seed.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        model_path = Path(args.model)
        train_classifier(args.input, model_path)
        print(f"Model saved: {model_path}")
        return 0

    if args.command == "analyze":
        model_path = Path(args.model)
        if args.auto_train and not model_path.exists():
            train_classifier(DEFAULT_TRAINING, model_path)
            print(f"Model trained: {model_path}")
        results = analyze_fields(
            input_csv=args.input,
            catalog_csv=args.catalog,
            model_path=model_path,
            output_csv=args.output_csv,
            output_json=args.output_json,
            grader_name=args.grader,
        )
        print(f"Analyzed fields: {len(results)}")
        print(f"CSV report: {args.output_csv}")
        print(f"JSON report: {args.output_json}")
        return 0

    if args.command == "evaluate":
        result = evaluate_classifier(
            training_csv=args.input,
            catalog_csv=args.catalog,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        summary = result["summary"]
        print(f"Samples: {summary['sample_count']}  Train: {summary['train_count']}  Test: {summary['test_count']}")
        print(f"Accuracy: {summary['accuracy']}  Macro-F1: {summary['macro_f1']}")
        print(f"Metrics: {result['metrics_path']}")
        print(f"Predictions: {result['predictions_path']}")
        print(f"Confusion matrix: {result['confusion_path']}")
        print(f"Workbook: {result['workbook_path']}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
