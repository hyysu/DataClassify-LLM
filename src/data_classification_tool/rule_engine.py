"""Risk rule matching for classification and grading 2.0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_classification_tool.models import ClassificationResult, FieldRecord


def _split_pipe(value: object) -> list[str]:
    text = str(value or "").strip()
    return [item.strip() for item in text.split("|") if item.strip()]


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


@dataclass(frozen=True)
class RiskRule:
    """One structured risk rule loaded from the rule catalog."""

    rule_id: str
    rule_name: str
    priority: int
    enabled: bool
    keywords: list[str]
    regex_features: list[str]
    output_category_level1: str
    output_category_level2: str
    minimum_level: str
    risk_tags: list[str]
    legal_basis: list[str]
    explanation: str
    controls: list[str]
    review_required: bool


@dataclass(frozen=True)
class RiskRuleMatch:
    """A rule matched against one field."""

    rule: RiskRule
    matched_by: list[str]


class RiskRuleEngine:
    """Match field metadata and classifier evidence against risk rules."""

    def __init__(self, rules: list[RiskRule]) -> None:
        self.rules = [rule for rule in rules if rule.enabled]

    @classmethod
    def from_csv(cls, path: str | Path) -> "RiskRuleEngine":
        frame = pd.read_csv(path).fillna("")
        rules: list[RiskRule] = []
        for row in frame.to_dict(orient="records"):
            rules.append(
                RiskRule(
                    rule_id=str(row["rule_id"]).strip(),
                    rule_name=str(row["rule_name"]).strip(),
                    priority=int(row.get("priority", 0) or 0),
                    enabled=_to_bool(row.get("enabled", "true")),
                    keywords=_split_pipe(row.get("keywords", "")),
                    regex_features=_split_pipe(row.get("regex_features", "")),
                    output_category_level1=str(row.get("output_category_level1", "")).strip(),
                    output_category_level2=str(row.get("output_category_level2", "")).strip(),
                    minimum_level=str(row.get("minimum_level", "L2")).strip(),
                    risk_tags=_split_pipe(row.get("risk_tags", "")),
                    legal_basis=_split_pipe(row.get("legal_basis", "")),
                    explanation=str(row.get("explanation", "")).strip(),
                    controls=_split_pipe(row.get("controls", "")),
                    review_required=_to_bool(row.get("review_required", "false")),
                )
            )
        return cls(rules)

    def match(self, record: FieldRecord, classification: ClassificationResult) -> list[RiskRuleMatch]:
        text_parts = [
            record.table_name,
            record.table_comment,
            record.column_name,
            record.column_comment,
            record.data_type,
            record.business_context,
            record.sample_value,
            " ".join(record.samples()),
        ]
        haystack = " ".join(text_parts).lower()
        pattern_set = {item.lower() for item in classification.matched_patterns}
        pattern_set.update(str(item).lower() for item in classification.sample_profile.get("regex_features", []))

        matches: list[RiskRuleMatch] = []
        for rule in self.rules:
            matched_by: list[str] = []
            for keyword in rule.keywords:
                if keyword.lower() in haystack:
                    matched_by.append(f"keyword:{keyword}")
            for regex_feature in rule.regex_features:
                if regex_feature.lower() in pattern_set or regex_feature.lower() in haystack:
                    matched_by.append(f"regex:{regex_feature}")
            if matched_by:
                matches.append(RiskRuleMatch(rule=rule, matched_by=matched_by))

        return sorted(matches, key=lambda item: item.rule.priority, reverse=True)
