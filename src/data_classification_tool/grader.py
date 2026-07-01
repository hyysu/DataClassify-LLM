"""Data security grading logic."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from data_classification_tool.catalog import get_rule
from data_classification_tool.models import ClassificationResult, FieldRecord, GradingResult, LabelRule


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


class RuleBasedGrader:
    """Local grading engine used before an LLM API key is available."""

    def __init__(self, catalog: dict[str, LabelRule], review_threshold: float = 0.55) -> None:
        self.catalog = catalog
        self.review_threshold = review_threshold

    def grade(self, record: FieldRecord, classification: ClassificationResult) -> GradingResult:
        """Grade a field from catalog rules and deterministic signals."""

        rule = get_rule(self.catalog, classification.label_id)
        level = rule.default_level or "L2"
        reasons = [f"字段被识别为“{rule.category}”。"]

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

        requires_review = classification.confidence < self.review_threshold or classification.label_id == "unknown"
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
            controls=rule.controls,
            requires_review=requires_review,
            source="rule",
        )


class LLMGrader:
    """LLM grading adapter. It is intentionally optional for this MVP."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

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

    def grade(self, record: FieldRecord, classification: ClassificationResult) -> GradingResult:
        """Call an OpenAI-compatible LLM API when a key is configured."""

        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not configured. Use --grader rule for local tests.")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是数据安全分类分级专家。"},
                {"role": "user", "content": self.build_prompt(record, classification)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return GradingResult(
            level=data["level"],
            level_name=data.get("level_name", LEVEL_NAMES.get(data["level"], data["level"])),
            reason=data.get("reason", ""),
            confidence=float(data.get("confidence", 0.0)),
            controls=list(data.get("controls", [])),
            requires_review=bool(data.get("requires_review", False)),
            source="llm",
        )

