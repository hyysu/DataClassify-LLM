"""Compatibility wrapper for the field-level Bayesian classifier."""

from __future__ import annotations

from pathlib import Path

from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.feature_extractor import split_identifier
from data_classification_tool.models import (
    ClassificationCandidate,
    ClassificationResult,
    FieldRecord,
    LabelRule,
)


PATTERN_LABEL_BOOSTS = {
    "mobile_phone": {"mobile_phone": 1.4},
    "email": {"email": 1.4},
    "id_number": {"id_number": 1.6},
    "bank_card": {"bank_card": 1.6},
    "ip_address": {"ip_address": 1.2},
    "mac_address": {"device_id": 1.4},
    "location": {"precise_location": 1.2},
    "secret_token": {"auth_token": 1.4, "password": 0.6},
    "amount": {"transaction_amount": 1.2},
}


def _is_ascii(text: str) -> bool:
    return all(ord(char) < 128 for char in text)


def _record_text(record: FieldRecord) -> str:
    return " ".join(
        [
            record.table_name,
            record.table_comment,
            record.column_name,
            record.column_comment,
            record.data_type,
            record.business_context,
        ]
    ).lower()


def _record_tokens(record: FieldRecord) -> set[str]:
    tokens = split_identifier(record.table_name) + split_identifier(record.column_name)
    return {token.lower() for token in tokens}


def _keyword_matches(keyword: str, text: str, tokens: set[str]) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    if _is_ascii(keyword):
        if "_" in keyword:
            return keyword in text
        return keyword in tokens
    return keyword in text


def _keyword_weight(keyword: str) -> float:
    keyword = keyword.strip()
    if not keyword:
        return 0.0
    if _is_ascii(keyword):
        return 1.3 if "_" in keyword else 1.0
    if len(keyword) >= 4:
        return 1.6
    if len(keyword) >= 3:
        return 1.3
    return 1.0


def _catalog_boosts(record: FieldRecord, catalog: dict[str, LabelRule], patterns: list[str]) -> dict[str, float]:
    text = _record_text(record)
    tokens = _record_tokens(record)
    boosts = {label_id: 0.0 for label_id in catalog}
    for label_id, rule in catalog.items():
        for keyword in rule.keywords:
            if _keyword_matches(keyword, text, tokens):
                boosts[label_id] += _keyword_weight(keyword)
    for pattern in patterns:
        for label_id, boost in PATTERN_LABEL_BOOSTS.get(pattern, {}).items():
            boosts[label_id] = boosts.get(label_id, 0.0) + boost
    return boosts


class BayesianFieldClassifier:
    """Backward-compatible wrapper around DataFieldBayesClassifier."""

    def __init__(self, classifier: DataFieldBayesClassifier | None = None) -> None:
        self.classifier = classifier or DataFieldBayesClassifier()

    @property
    def pipeline(self):
        return self.classifier.pipeline

    @property
    def classes_(self):
        return self.classifier.pipeline.classes_

    def fit(self, records: list[FieldRecord], labels: list[str]) -> "BayesianFieldClassifier":
        """Train the Bayesian classifier."""

        self.classifier.fit(records, labels)
        return self

    def predict_one(
        self,
        record: FieldRecord,
        catalog: dict[str, LabelRule],
        top_n: int = 3,
    ) -> ClassificationResult:
        """Predict one field label with explainable top candidates."""

        base_prediction = self.classifier.predict(record, top_k=top_n)
        base_probabilities = self.classifier.predict_proba(record)
        boosts = _catalog_boosts(record, catalog, base_prediction.matched_patterns)
        adjusted_scores = {
            label: probability + boosts.get(label, 0.0)
            for label, probability in base_probabilities.items()
        }
        total_score = sum(adjusted_scores.values()) or 1.0
        ranked = sorted(
            ((label, score / total_score) for label, score in adjusted_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        predicted_label, confidence = ranked[0]
        rule = catalog.get(predicted_label)
        category = rule.category if rule else predicted_label
        candidates = [
            ClassificationCandidate(
                label_id=label,
                probability=round(float(probability), 4),
            )
            for label, probability in ranked[:top_n]
        ]
        return ClassificationResult(
            label_id=predicted_label,
            category=category,
            confidence=round(float(confidence), 4),
            candidates=candidates,
            matched_patterns=base_prediction.matched_patterns,
            need_review=float(confidence) < self.classifier.review_threshold,
            evidence_features=base_prediction.evidence_features,
            feature_text=base_prediction.feature_text,
            sample_profile=base_prediction.sample_profile,
        )

    def save(self, path: str | Path) -> None:
        """Persist the trained model."""

        self.classifier.save_model(path)

    @classmethod
    def load(cls, path: str | Path) -> "BayesianFieldClassifier":
        """Load a trained model."""

        return cls(classifier=DataFieldBayesClassifier.load_model(path))
