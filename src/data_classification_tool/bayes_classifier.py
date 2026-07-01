"""Naive Bayes classifier tailored for database field classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_classification_tool.feature_extractor import FieldFeatureExtractor
from data_classification_tool.models import FieldRecord


@dataclass(frozen=True)
class CategoryCandidate:
    """One predicted category candidate."""

    category: str
    probability: float


@dataclass(frozen=True)
class FieldPrediction:
    """Explainable prediction result for one field."""

    field: str
    predicted_category: str
    confidence: float
    top_candidates: list[CategoryCandidate] = field(default_factory=list)
    need_review: bool = False
    evidence_features: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    feature_text: str = ""
    sample_profile: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_from_any(value: FieldRecord | dict[str, Any]) -> FieldRecord:
    if isinstance(value, FieldRecord):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported record type: {type(value)!r}")
    sample_values = value.get("sample_values", ())
    if isinstance(sample_values, list):
        sample_values = tuple(str(item) for item in sample_values)
    elif isinstance(sample_values, str):
        sample_values = (sample_values,) if sample_values else ()
    return FieldRecord(
        table_name=str(value.get("table_name", "")),
        table_comment=str(value.get("table_comment", "")),
        column_name=str(value.get("column_name", "")),
        column_comment=str(value.get("column_comment", "")),
        data_type=str(value.get("data_type", "")),
        sample_value=str(value.get("sample_value", "")),
        sample_values=sample_values,
        business_context=str(value.get("business_context", "")),
        nullable=value.get("nullable"),
        is_primary_key=value.get("is_primary_key"),
        is_unique=value.get("is_unique"),
    )


def _label_from_any(value: FieldRecord | dict[str, Any]) -> str:
    if isinstance(value, dict):
        return str(value.get("label", value.get("label_id", ""))).strip()
    return str(getattr(value, "label", "")).strip()


class DataFieldBayesClassifier:
    """Naive Bayes classifier for database field metadata."""

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        feature_extractor: FieldFeatureExtractor | None = None,
        review_threshold: float = 0.6,
        default_top_k: int = 3,
    ) -> None:
        self.pipeline = pipeline or Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", MultinomialNB(alpha=0.05)),
            ]
        )
        self.feature_extractor = feature_extractor or FieldFeatureExtractor()
        self.review_threshold = review_threshold
        self.default_top_k = default_top_k

    def fit(
        self,
        train_records: Iterable[FieldRecord | dict[str, Any]],
        labels: Iterable[str] | None = None,
    ) -> "DataFieldBayesClassifier":
        """Train the classifier.

        If labels is omitted, each record must contain a ``label`` or
        ``label_id`` key.
        """

        raw_records = list(train_records)
        records = [_record_from_any(record) for record in raw_records]
        if labels is None:
            label_list = [_label_from_any(record) for record in raw_records]
        else:
            label_list = [str(label).strip() for label in labels]

        if len(records) != len(label_list):
            raise ValueError("records and labels must have the same length")
        if not all(label_list):
            raise ValueError("all training records must have a non-empty label")

        feature_texts = [self.feature_extractor.extract(record).feature_text for record in records]
        self.pipeline.fit(feature_texts, label_list)
        return self

    def predict(self, record: FieldRecord | dict[str, Any], top_k: int | None = None) -> FieldPrediction:
        """Predict one field and return explainable output."""

        field_record = _record_from_any(record)
        extracted = self.feature_extractor.extract(field_record)
        probabilities = self.predict_proba(field_record)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_n = top_k or self.default_top_k
        predicted_category, confidence = ranked[0]
        candidates = [
            CategoryCandidate(category=category, probability=round(float(probability), 4))
            for category, probability in ranked[:top_n]
        ]
        return FieldPrediction(
            field=field_record.column_name,
            predicted_category=predicted_category,
            confidence=round(float(confidence), 4),
            top_candidates=candidates,
            need_review=float(confidence) < self.review_threshold,
            evidence_features=extracted.evidence_features,
            matched_patterns=extracted.regex_features,
            feature_text=extracted.feature_text,
            sample_profile=extracted.sample_profile,
        )

    def predict_batch(
        self,
        records: Iterable[FieldRecord | dict[str, Any]],
        top_k: int | None = None,
    ) -> list[FieldPrediction]:
        """Predict a batch of fields."""

        return [self.predict(record, top_k=top_k) for record in records]

    def predict_proba(self, record: FieldRecord | dict[str, Any]) -> dict[str, float]:
        """Return category probability mapping for one field."""

        field_record = _record_from_any(record)
        feature_text = self.feature_extractor.extract(field_record).feature_text
        probabilities = self.pipeline.predict_proba([feature_text])[0]
        labels = list(self.pipeline.classes_)
        return {
            label: round(float(probability), 6)
            for label, probability in zip(labels, probabilities)
        }

    def to_llm_grading_input(
        self,
        record: FieldRecord | dict[str, Any],
        prediction: FieldPrediction | None = None,
    ) -> dict[str, Any]:
        """Convert a Bayes prediction into the payload expected by LLM grading."""

        field_record = _record_from_any(record)
        prediction = prediction or self.predict(field_record)
        return {
            "field_name": field_record.column_name,
            "field_comment": field_record.column_comment,
            "table_name": field_record.table_name,
            "table_comment": field_record.table_comment,
            "data_type": field_record.data_type,
            "bayes_category": prediction.predicted_category,
            "bayes_confidence": prediction.confidence,
            "evidence_features": prediction.evidence_features,
            "matched_patterns": prediction.matched_patterns,
            "sample_profile": prediction.sample_profile,
            "need_review": prediction.need_review,
        }

    def save_model(self, path: str | Path) -> None:
        """Persist the model and lightweight runtime settings."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "review_threshold": self.review_threshold,
                "default_top_k": self.default_top_k,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str | Path) -> "DataFieldBayesClassifier":
        """Load a saved model.

        Older joblib files containing only a sklearn Pipeline are also
        supported.
        """

        payload = joblib.load(path)
        if isinstance(payload, Pipeline):
            return cls(pipeline=payload)
        return cls(
            pipeline=payload["pipeline"],
            review_threshold=payload.get("review_threshold", 0.6),
            default_top_k=payload.get("default_top_k", 3),
        )
