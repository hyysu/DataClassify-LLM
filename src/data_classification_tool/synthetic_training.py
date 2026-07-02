"""Iterative training utilities for synthetic field metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.synthetic_data import generate_synthetic_training_records


@dataclass(frozen=True)
class EvaluationMetrics:
    """Simple classification metrics for generated records."""

    accuracy: float
    total: int
    correct: int
    per_label_accuracy: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRoundMetrics:
    """Validation outcome for one training round."""

    round_index: int
    records_per_category: int
    train_records: int
    validation_accuracy: float
    validation_sets: int


@dataclass(frozen=True)
class IterativeTrainingResult:
    """Final result of iterative synthetic training."""

    reached_target: bool
    best_round: int
    best_validation_accuracy: float
    final_test_accuracy: float
    model_path: str
    history: list[TrainingRoundMetrics]


def evaluate_classifier(
    classifier: DataFieldBayesClassifier,
    records: list[dict[str, object]],
) -> EvaluationMetrics:
    """Evaluate classifier on labeled synthetic records."""

    total = len(records)
    correct = 0
    label_totals: dict[str, int] = {}
    label_correct: dict[str, int] = {}

    for record in records:
        expected = str(record["label"])
        prediction = classifier.predict(record)
        label_totals[expected] = label_totals.get(expected, 0) + 1
        if prediction.predicted_category == expected:
            correct += 1
            label_correct[expected] = label_correct.get(expected, 0) + 1

    per_label_accuracy = {
        label: round(label_correct.get(label, 0) / count, 4)
        for label, count in sorted(label_totals.items())
    }
    return EvaluationMetrics(
        accuracy=round(correct / total, 4) if total else 0.0,
        total=total,
        correct=correct,
        per_label_accuracy=per_label_accuracy,
    )


def evaluate_on_generated_sets(
    classifier: DataFieldBayesClassifier,
    records_per_category: int,
    sample_values_per_record: int,
    seeds: list[int],
) -> EvaluationMetrics:
    """Evaluate on multiple generated validation sets and return aggregate metrics."""

    records: list[dict[str, object]] = []
    for seed in seeds:
        records.extend(
            generate_synthetic_training_records(
                records_per_category=records_per_category,
                seed=seed,
                sample_values_per_record=sample_values_per_record,
            )
        )
    return evaluate_classifier(classifier, records)


def train_synthetic_until_target(
    model_path: str | Path,
    target_accuracy: float = 0.92,
    max_rounds: int = 8,
    initial_records_per_category: int = 10,
    records_increment: int = 10,
    eval_interval: int = 1,
    train_seed: int = 42,
    validation_seed: int = 2024,
    validation_seed_count: int = 3,
    validation_records_per_category: int = 8,
    test_seed: int = 9001,
    test_records_per_category: int = 12,
    sample_values_per_record: int = 3,
) -> IterativeTrainingResult:
    """Train repeatedly on generated data until validation accuracy reaches target.

    Validation sets are used for stopping decisions. The final test set is
    generated from a different seed and evaluated only after training stops.
    """

    if not 0 < target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1]")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if records_increment <= 0:
        raise ValueError("records_increment must be positive")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if validation_seed_count <= 0:
        raise ValueError("validation_seed_count must be positive")

    model_path = Path(model_path)
    history: list[TrainingRoundMetrics] = []
    best_classifier: DataFieldBayesClassifier | None = None
    best_round = 0
    best_validation_accuracy = -1.0
    reached_target = False

    validation_seeds = [validation_seed + index for index in range(validation_seed_count)]

    for round_index in range(1, max_rounds + 1):
        records_per_category = initial_records_per_category + (round_index - 1) * records_increment
        train_records = generate_synthetic_training_records(
            records_per_category=records_per_category,
            seed=train_seed + round_index,
            sample_values_per_record=sample_values_per_record,
        )
        classifier = DataFieldBayesClassifier().fit(train_records)

        if round_index % eval_interval != 0:
            continue

        validation_metrics = evaluate_on_generated_sets(
            classifier=classifier,
            records_per_category=validation_records_per_category,
            sample_values_per_record=sample_values_per_record,
            seeds=validation_seeds,
        )
        history.append(
            TrainingRoundMetrics(
                round_index=round_index,
                records_per_category=records_per_category,
                train_records=len(train_records),
                validation_accuracy=validation_metrics.accuracy,
                validation_sets=len(validation_seeds),
            )
        )

        if validation_metrics.accuracy > best_validation_accuracy:
            best_classifier = classifier
            best_round = round_index
            best_validation_accuracy = validation_metrics.accuracy

        if validation_metrics.accuracy >= target_accuracy:
            reached_target = True
            break

    if best_classifier is None:
        raise RuntimeError("No validation round was executed. Check eval_interval and max_rounds.")

    test_records = generate_synthetic_training_records(
        records_per_category=test_records_per_category,
        seed=test_seed,
        sample_values_per_record=sample_values_per_record,
    )
    test_metrics = evaluate_classifier(best_classifier, test_records)
    best_classifier.save_model(model_path)

    return IterativeTrainingResult(
        reached_target=reached_target,
        best_round=best_round,
        best_validation_accuracy=round(best_validation_accuracy, 4),
        final_test_accuracy=test_metrics.accuracy,
        model_path=str(model_path),
        history=history,
    )

