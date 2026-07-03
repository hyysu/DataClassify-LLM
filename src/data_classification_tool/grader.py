"""Data security grading logic."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from data_classification_tool.catalog import get_rule
from data_classification_tool.config import DEFAULT_RISK_RULE_CSV
from data_classification_tool.models import ClassificationResult, FieldRecord, GradingResult, LabelRule
from data_classification_tool.rule_engine import RiskRuleEngine


LEVEL_NAMES = {
    "L1": "公开或低敏数据",
    "L2": "内部一般数据",
    "L3": "敏感数据",
    "L4": "高敏感或重要数据",
}

LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}


def max_level(*levels: str) -> str:
    """Return the highest level from a group of level codes."""

    return max(levels, key=lambda level: LEVEL_RANK.get(level, 0))


def _merge_unique(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item and item not in merged:
                merged.append(item)
    return merged


def merge_grading_results(rule_grading: GradingResult, llm_grading: GradingResult) -> GradingResult:
    """Merge deterministic rule grading with LLM advice without weakening strong rules."""

    final_level = max_level(rule_grading.level, llm_grading.level)
    llm_downgraded_rule = LEVEL_RANK.get(llm_grading.level, 0) < LEVEL_RANK.get(rule_grading.level, 0)
    reasons = [
        f"规则结论：{rule_grading.reason}",
        f"LLM建议：{llm_grading.reason}",
    ]
    if llm_downgraded_rule:
        reasons.append("LLM建议低于规则库保底等级，最终按规则库保底。")

    return GradingResult(
        level=final_level,
        level_name=LEVEL_NAMES.get(final_level, final_level),
        reason="".join(reasons),
        confidence=max(rule_grading.confidence, llm_grading.confidence),
        controls=_merge_unique(rule_grading.controls, llm_grading.controls),
        requires_review=rule_grading.requires_review or llm_grading.requires_review or llm_downgraded_rule,
        source="rule+llm",
        risk_tags=_merge_unique(rule_grading.risk_tags, llm_grading.risk_tags),
        matched_rules=_merge_unique(rule_grading.matched_rules, llm_grading.matched_rules),
        legal_basis=_merge_unique(rule_grading.legal_basis, llm_grading.legal_basis),
        rule_category_level1=rule_grading.rule_category_level1,
        rule_category_level2=rule_grading.rule_category_level2,
    )


class RuleBasedGrader:
    """Local grading engine used before an LLM API key is available."""

    def __init__(
        self,
        catalog: dict[str, LabelRule],
        review_threshold: float = 0.55,
        risk_rule_csv: str | Path | None = DEFAULT_RISK_RULE_CSV,
    ) -> None:
        self.catalog = catalog
        self.review_threshold = review_threshold
        self.risk_rule_engine = (
            RiskRuleEngine.from_csv(risk_rule_csv)
            if risk_rule_csv and Path(risk_rule_csv).exists()
            else None
        )

    def grade(self, record: FieldRecord, classification: ClassificationResult) -> GradingResult:
        """Grade a field from catalog rules and deterministic signals."""

        rule = get_rule(self.catalog, classification.label_id)
        level = rule.default_level or "L2"
        reasons = [f"字段被识别为“{rule.category}”。"]
        controls = list(rule.controls)
        risk_tags: list[str] = []
        matched_rules: list[str] = []
        legal_basis: list[str] = []
        rule_category_level1 = ""
        rule_category_level2 = ""

        if rule.is_sensitive_personal_info:
            level = max_level(level, "L4")
            reasons.append("该类别通常属于敏感个人信息，泄露后可能造成较高个人权益风险。")
        elif rule.is_personal_info:
            level = max_level(level, "L3")
            reasons.append("该类别通常属于个人信息，需要采取访问控制和脱敏等保护措施。")

        if rule.important_data_hint:
            level = max_level(level, "L3")
            reasons.append("该类别可能涉及企业经营敏感信息或重要数据线索，建议结合业务范围复核。")

        if any(pattern in classification.matched_patterns for pattern in ["looks_like_secret", "looks_like_id_number"]):
            level = max_level(level, "L4")
            reasons.append("字段命中了高敏感模式特征。")

        if self.risk_rule_engine:
            risk_matches = self.risk_rule_engine.match(record, classification)
            for match in risk_matches:
                risk_rule = match.rule
                level = max_level(level, risk_rule.minimum_level)
                matched_rules.append(risk_rule.rule_id)
                risk_tags = _merge_unique(risk_tags, risk_rule.risk_tags)
                legal_basis = _merge_unique(legal_basis, risk_rule.legal_basis)
                controls = _merge_unique(controls, risk_rule.controls)
                if not rule_category_level1:
                    rule_category_level1 = risk_rule.output_category_level1
                    rule_category_level2 = risk_rule.output_category_level2
                reasons.append(f"命中规则 {risk_rule.rule_id}：{risk_rule.explanation}")

        requires_review = classification.confidence < self.review_threshold or classification.label_id == "unknown"
        if any(tag in risk_tags for tag in ["important_data_candidate", "core_data_candidate"]):
            requires_review = True
            reasons.append("命中重要数据或核心数据候选标记，需要人工复核。")
        if self.risk_rule_engine:
            requires_review = requires_review or any(
                match.rule.review_required for match in self.risk_rule_engine.match(record, classification)
            )
        if requires_review:
            reasons.append("模型置信度偏低或类别不明确，需要人工复核。")

        confidence = classification.confidence
        if requires_review:
            confidence = min(confidence, 0.5)

        return GradingResult(
            level=level,
            level_name=LEVEL_NAMES.get(level, level),
            reason="".join(reasons),
            confidence=round(float(confidence), 4),
            controls=controls,
            requires_review=requires_review,
            source="rule",
            risk_tags=risk_tags,
            matched_rules=matched_rules,
            legal_basis=legal_basis,
            rule_category_level1=rule_category_level1,
            rule_category_level2=rule_category_level2,
        )


class LLMGrader:
    """LLM grading adapter. It is intentionally optional for this MVP."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    def _is_ollama(self) -> bool:
        return "11434" in self.base_url or "ollama" in self.base_url.lower()

    def _ollama_api_url(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1") + "/api/chat"

    def build_prompt(self, record: FieldRecord, classification: ClassificationResult) -> str:
        """Build a strict JSON-oriented grading prompt."""

        payload = {
            "field": asdict(record),
            "classification": asdict(classification),
            "grading_levels": LEVEL_NAMES,
            "requirements": [
                "根据字段语义、分类标签、样例摘要和业务场景判断数据安全等级。",
                "不要要求输入真实个人信息样例。",
                "如果无法判断，requires_review 必须为 true。",
                "只输出 JSON，不输出 Markdown。",
            ],
            "output_schema": {
                "level": "L1/L2/L3/L4",
                "level_name": "等级名称",
                "reason": "判断理由",
                "confidence": "0 到 1 的数字",
                "controls": ["建议控制措施"],
                "requires_review": "布尔值",
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_messages(self, record: FieldRecord, classification: ClassificationResult) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "你是数据安全分类分级专家。只输出合法 JSON，不输出 Markdown。"},
            {"role": "user", "content": self.build_prompt(record, classification)},
        ]

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            content = content.removeprefix("json").strip()
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end >= start:
            content = content[start : end + 1]
        return json.loads(content)

    def _normalize_result(self, data: dict[str, Any]) -> GradingResult:
        level = str(data.get("level", "L2")).upper()
        if level not in LEVEL_NAMES:
            level = "L2"
        controls = data.get("controls", [])
        if isinstance(controls, str):
            controls = [controls]
        return GradingResult(
            level=level,
            level_name=str(data.get("level_name") or LEVEL_NAMES[level]),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
            controls=list(controls),
            requires_review=bool(data.get("requires_review", False)),
            source="llm",
            risk_tags=[],
            matched_rules=[],
            legal_basis=[],
            rule_category_level1="",
            rule_category_level2="",
        )

    def _grade_with_ollama(self, record: FieldRecord, classification: ClassificationResult) -> GradingResult:
        import requests

        response = requests.post(
            self._ollama_api_url(),
            json={
                "model": self.model,
                "messages": self._build_messages(record, classification),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "{}")
        return self._normalize_result(self._parse_json_content(content))

    def grade(self, record: FieldRecord, classification: ClassificationResult) -> GradingResult:
        """Call an OpenAI-compatible LLM API when a key is configured."""

        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not configured. Use --grader rule for local tests.")

        if self._is_ollama():
            return self._grade_with_ollama(record, classification)

        from openai import OpenAI

        messages = self._build_messages(record, classification)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return self._normalize_result(self._parse_json_content(content))
