from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_classification_tool.synthetic_training import train_synthetic_until_target


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "synthetic_until_target_bayes.joblib"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "synthetic_training_history.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Iteratively train Bayes classifier on generated synthetic data.")
    parser.add_argument("--target-accuracy", type=float, default=0.92, help="Stop when validation accuracy reaches this value.")
    parser.add_argument("--max-rounds", type=int, default=8, help="Maximum training rounds.")
    parser.add_argument("--initial-records-per-category", type=int, default=10, help="Round-1 synthetic records per category.")
    parser.add_argument("--records-increment", type=int, default=10, help="Records per category added each round.")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N rounds.")
    parser.add_argument("--validation-seed-count", type=int, default=3, help="Number of generated validation sets.")
    parser.add_argument("--validation-records-per-category", type=int, default=8)
    parser.add_argument("--test-records-per-category", type=int, default=12)
    parser.add_argument("--samples-per-record", type=int, default=3)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=2024)
    parser.add_argument("--test-seed", type=int, default=9001)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Output model path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Training history JSON report.")
    args = parser.parse_args()

    result = train_synthetic_until_target(
        model_path=args.model,
        target_accuracy=args.target_accuracy,
        max_rounds=args.max_rounds,
        initial_records_per_category=args.initial_records_per_category,
        records_increment=args.records_increment,
        eval_interval=args.eval_interval,
        train_seed=args.train_seed,
        validation_seed=args.validation_seed,
        validation_seed_count=args.validation_seed_count,
        validation_records_per_category=args.validation_records_per_category,
        test_seed=args.test_seed,
        test_records_per_category=args.test_records_per_category,
        sample_values_per_record=args.samples_per_record,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

