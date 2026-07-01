"""Domain models used by the classification pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldRecord:
    """Metadata for one database or file field."""

    table_name: str
    column_name: str
    table_comment: str = ""
    column_comment: str = ""
    data_type: str = ""
    sample_value: str = ""
    sample_values: tuple[str, ...] = field(default_factory=tuple)
    business_context: str = ""
    nullable: bool | None = None
    is_primary_key: bool | None = None
    is_unique: bool | None = None

    def samples(self) -> tuple[str, ...]:
        """Return all available sample values without changing the original record."""

        if self.sample_values:
            return self.sample_values
        if self.sample_value:
            return (self.sample_value,)
        return ()


@dataclass(frozen=True)
class LabelRule:
    """One label definition from the field label catalog."""

    label_id: str
    category: str
    description: str
    default_level: str
    is_personal_info: bool
    is_sensitive_personal_info: bool
    important_data_hint: bool
    controls: list[str]
    keywords: list[str]


@dataclass(frozen=True)
class ClassificationCandidate:
    """One candidate label predicted by the classifier."""

    label_id: str
    probability: float


@dataclass(frozen=True)
class ClassificationResult:
    """Bayesian classification output for one field."""

    label_id: str
    category: str
    confidence: float
    candidates: list[ClassificationCandidate] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    need_review: bool = False
    evidence_features: list[str] = field(default_factory=list)
    feature_text: str = ""
    sample_profile: dict[str, Any] = field(default_factory=dict)

    def to_llm_payload(self, field: FieldRecord) -> dict[str, Any]:
        """Create the minimal payload expected by an LLM grading module."""

        return {
            "field_name": field.column_name,
            "field_comment": field.column_comment,
            "table_name": field.table_name,
            "table_comment": field.table_comment,
            "data_type": field.data_type,
            "bayes_category": self.category,
            "bayes_label_id": self.label_id,
            "bayes_confidence": self.confidence,
            "evidence_features": self.evidence_features,
            "matched_patterns": self.matched_patterns,
            "sample_profile": self.sample_profile,
            "need_review": self.need_review,
        }


@dataclass(frozen=True)
class GradingResult:
    """Data security grading output for one field."""

    level: str
    level_name: str
    reason: str
    confidence: float
    controls: list[str]
    requires_review: bool
    source: str = "rule"


@dataclass(frozen=True)
class AnalysisResult:
    """Complete output for one field."""

    field: FieldRecord
    classification: ClassificationResult
    grading: GradingResult

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses to a serializable dictionary."""

        return asdict(self)
