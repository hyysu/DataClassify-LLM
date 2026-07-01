"""Field-level feature extraction for Naive Bayes classification."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from data_classification_tool.models import FieldRecord
from data_classification_tool.regex_rules import (
    build_sample_profile,
    detect_regex_features,
    evidence_from_regex_features,
    regex_weight_tokens,
    sample_profile_tokens,
)

try:  # pragma: no cover - fallback exists for environments without jieba.
    import jieba
except ImportError:  # pragma: no cover
    jieba = None


DATA_TYPE_ALIASES = {
    "char": "type_string",
    "varchar": "type_string",
    "nvarchar": "type_string",
    "text": "type_text",
    "json": "type_json",
    "int": "type_integer",
    "integer": "type_integer",
    "bigint": "type_integer",
    "decimal": "type_decimal",
    "number": "type_decimal",
    "numeric": "type_decimal",
    "float": "type_decimal",
    "double": "type_decimal",
    "date": "type_date",
    "datetime": "type_datetime",
    "timestamp": "type_datetime",
    "bool": "type_boolean",
    "boolean": "type_boolean",
    "binary": "type_binary",
    "blob": "type_binary",
}


@dataclass(frozen=True)
class ExtractedFieldFeatures:
    """Feature extraction result for one field."""

    feature_text: str
    evidence_features: list[str] = field(default_factory=list)
    regex_features: list[str] = field(default_factory=list)
    sample_profile: dict[str, object] = field(default_factory=dict)


def split_identifier(value: str) -> list[str]:
    """Split snake_case, kebab-case and camelCase identifiers."""

    if not value:
        return []
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    spaced = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", spaced)
    return [token.lower() for token in spaced.split() if token.strip()]


def tokenize_chinese(text: str) -> list[str]:
    """Tokenize Chinese text with jieba when available, otherwise use a safe fallback."""

    text = str(text or "").strip()
    if not text:
        return []
    if jieba is not None:
        return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]

    # Fallback: keep Chinese phrases and ASCII tokens.
    return [token.lower() for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text)]


def normalize_data_type(data_type: str) -> list[str]:
    """Convert database data types to stable feature tokens."""

    raw = str(data_type or "").strip().lower()
    if not raw:
        return []
    base = re.sub(r"\(.*\)", "", raw).strip()
    tokens = [f"raw_type_{base}"]
    for key, alias in DATA_TYPE_ALIASES.items():
        if key in base:
            tokens.append(alias)
            break
    return tokens


def bool_feature(name: str, value: bool | None) -> list[str]:
    if value is None:
        return []
    return [f"{name}_{str(value).lower()}"]


class FieldFeatureExtractor:
    """Build safe, field-aware text features for Naive Bayes.

    The extractor never appends raw sample values to feature_text. It only uses
    derived profile tokens and regex match indicators.
    """

    def __init__(self, strong_feature_weight: int = 3) -> None:
        self.strong_feature_weight = strong_feature_weight

    def extract(self, record: FieldRecord) -> ExtractedFieldFeatures:
        metadata_text = self._metadata_text(record)
        samples = record.samples()
        sample_profile = build_sample_profile(samples)
        regex_features = detect_regex_features(metadata_text, samples)

        tokens: list[str] = []
        evidence: list[str] = []

        tokens.extend(self._identifier_tokens(record.table_name, prefix="table"))
        tokens.extend(self._identifier_tokens(record.column_name, prefix="column"))
        tokens.extend(self._comment_tokens(record.table_comment, prefix="table_comment"))
        tokens.extend(self._comment_tokens(record.column_comment, prefix="column_comment"))
        tokens.extend(self._comment_tokens(record.business_context, prefix="business_context"))
        tokens.extend(normalize_data_type(record.data_type))
        tokens.extend(bool_feature("nullable", record.nullable))
        tokens.extend(bool_feature("primary_key", record.is_primary_key))
        tokens.extend(bool_feature("unique", record.is_unique))
        tokens.extend(sample_profile_tokens(sample_profile))
        tokens.extend(regex_weight_tokens(regex_features))

        evidence.extend(evidence_from_regex_features(regex_features))
        evidence.extend(self._human_evidence(record, sample_profile))

        feature_text = " ".join(token for token in tokens if token)
        return ExtractedFieldFeatures(
            feature_text=feature_text,
            evidence_features=sorted(set(evidence)),
            regex_features=regex_features,
            sample_profile=sample_profile,
        )

    def _metadata_text(self, record: FieldRecord) -> str:
        return " ".join(
            [
                record.table_name,
                record.table_comment,
                record.column_name,
                record.column_comment,
                record.data_type,
                record.business_context,
            ]
        )

    def _identifier_tokens(self, value: str, prefix: str) -> list[str]:
        parts = split_identifier(value)
        tokens = [str(value or "").lower(), *parts]
        return [f"{prefix}_{token}" for token in tokens if token]

    def _comment_tokens(self, text: str, prefix: str) -> list[str]:
        return [f"{prefix}_{token}" for token in tokenize_chinese(text)]

    def _human_evidence(self, record: FieldRecord, sample_profile: dict[str, object]) -> list[str]:
        evidence: list[str] = []
        if record.table_name:
            evidence.append(f"{record.table_name}表")
        if record.column_comment:
            evidence.append(record.column_comment)
        if record.is_primary_key:
            evidence.append("主键字段")
        if record.is_unique:
            evidence.append("唯一字段")
        if sample_profile.get("fixed_length"):
            evidence.append("样本固定长度")
        if sample_profile.get("high_unique_ratio"):
            evidence.append("样本高唯一率")

        avg_length = sample_profile.get("avg_length", 0)
        if avg_length:
            evidence.append(f"平均长度约{avg_length}")
        return evidence


DEFAULT_EXTRACTOR = FieldFeatureExtractor()


def build_feature_text(record: FieldRecord) -> str:
    """Compatibility helper used by existing code."""

    return DEFAULT_EXTRACTOR.extract(record).feature_text


def detect_patterns(record: FieldRecord) -> list[str]:
    """Compatibility helper returning regex feature names."""

    features = DEFAULT_EXTRACTOR.extract(record)
    return features.regex_features


def extract_records(records: Iterable[FieldRecord]) -> list[ExtractedFieldFeatures]:
    return [DEFAULT_EXTRACTOR.extract(record) for record in records]

