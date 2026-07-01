from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.config import DEFAULT_BAYES_MODEL_PATH, DEFAULT_FIELD_TRAINING_JSON
from data_classification_tool.io import read_field_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Train field-level Naive Bayes classifier.")
    parser.add_argument("--input", default=str(DEFAULT_FIELD_TRAINING_JSON), help="Training JSON/CSV path.")
    parser.add_argument("--model", default=str(DEFAULT_BAYES_MODEL_PATH), help="Model output path.")
    args = parser.parse_args()

    records, labels = read_field_records(args.input, require_label=True)
    classifier = DataFieldBayesClassifier().fit(records, labels)
    classifier.save_model(args.model)
    print(f"trained_records={len(records)}")
    print(f"model={args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

