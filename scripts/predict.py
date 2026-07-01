from pathlib import Path
import argparse
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.config import DEFAULT_BAYES_MODEL_PATH, DEFAULT_FIELD_PREDICT_JSON
from data_classification_tool.io import read_field_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict field categories with the Bayes classifier.")
    parser.add_argument("--input", default=str(DEFAULT_FIELD_PREDICT_JSON), help="Prediction JSON/CSV path.")
    parser.add_argument("--model", default=str(DEFAULT_BAYES_MODEL_PATH), help="Model path.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate categories.")
    args = parser.parse_args()

    records, _ = read_field_records(args.input, require_label=False)
    classifier = DataFieldBayesClassifier.load_model(args.model)
    predictions = classifier.predict_batch(records, top_k=args.top_k)
    print(json.dumps([prediction.to_dict() for prediction in predictions], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

