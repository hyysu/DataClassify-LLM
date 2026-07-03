"""End-to-end training and analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from data_classification_tool.catalog import load_label_catalog
from data_classification_tool.classifier import BayesianFieldClassifier
from data_classification_tool.grader import LLMGrader, RuleBasedGrader, merge_grading_results
from data_classification_tool.io import read_field_records, write_results_csv, write_results_json
from data_classification_tool.models import AnalysisResult


def train_classifier(training_csv: str | Path, model_path: str | Path) -> BayesianFieldClassifier:
    """Train and save the Bayesian field classifier."""

    records, labels = read_field_records(training_csv, require_label=True)
    classifier = BayesianFieldClassifier().fit(records, labels)
    classifier.save(model_path)
    return classifier


def analyze_fields(
    input_csv: str | Path,
    catalog_csv: str | Path,
    model_path: str | Path,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
    grader_name: str = "rule",
) -> list[AnalysisResult]:
    """Run classification and grading for field metadata."""

    catalog = load_label_catalog(catalog_csv)
    classifier = BayesianFieldClassifier.load(model_path)
    records, _ = read_field_records(input_csv, require_label=False)

    rule_grader = RuleBasedGrader(catalog)
    llm_grader = LLMGrader() if grader_name == "llm" else None
    if grader_name not in {"rule", "llm"}:
        raise ValueError(f"Unsupported grader: {grader_name}")

    results: list[AnalysisResult] = []
    for record in records:
        classification = classifier.predict_one(record, catalog)
        rule_grading = rule_grader.grade(record, classification)
        if llm_grader:
            llm_grading = llm_grader.grade(record, classification)
            grading = merge_grading_results(rule_grading, llm_grading)
        else:
            grading = rule_grading
        results.append(AnalysisResult(field=record, classification=classification, grading=grading))

    if output_csv:
        write_results_csv(results, output_csv)
    if output_json:
        write_results_json(results, output_json)

    return results
