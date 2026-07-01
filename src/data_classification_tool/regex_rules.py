"""Regex and sample profiling rules for database field classification."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
import re
from typing import Iterable


@dataclass(frozen=True)
class RegexRule:
    """A named regex rule with an evidence label and optional strong weight."""

    name: str
    pattern: re.Pattern[str]
    evidence: str
    weight: int = 1


REGEX_RULES: tuple[RegexRule, ...] = (
    RegexRule("mobile_phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)|1[3-9]\d\*{4}\d{4}"), "手机号正则匹配", 5),
    RegexRule("email", re.compile(r"[A-Za-z0-9._%+-]+@[\w.-]+\.[A-Za-z]{2,}"), "邮箱正则匹配", 5),
    RegexRule("id_number", re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{6}\*{8,}\d{3,4})(?!\d)"), "身份证正则匹配", 6),
    RegexRule("bank_card", re.compile(r"(?<!\d)(\d{13,19}|\d{4}\*{4,}\d{4})(?!\d)"), "银行卡号正则匹配", 6),
    RegexRule("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP地址正则匹配", 4),
    RegexRule("url", re.compile(r"https?://[\w./?=&%:-]+|www\.[\w./?=&%:-]+", re.I), "URL正则匹配", 3),
    RegexRule("date", re.compile(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?\b"), "日期格式匹配", 2),
    RegexRule("datetime", re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(:\d{2})?\b"), "日期时间格式匹配", 2),
    RegexRule("amount", re.compile(r"^-?\d+(\.\d{1,4})?$"), "金额数值格式", 2),
    RegexRule("social_credit_code", re.compile(r"\b[0-9A-Z]{18}\b"), "统一社会信用代码正则匹配", 5),
    RegexRule("mac_address", re.compile(r"\b[0-9A-Fa-f]{2}([:-][0-9A-Fa-f*]{2}){5}\b"), "MAC地址正则匹配", 4),
    RegexRule("secret_token", re.compile(r"(sk-|token_|eyJ|AKIA|api[_-]?key|secret|session|passwd|password)", re.I), "密钥令牌格式特征", 5),
    RegexRule("location", re.compile(r"(lat|lng|gps|longitude|latitude|经度|纬度|定位|轨迹)", re.I), "位置字段特征", 3),
)


def detect_regex_features(text: str, samples: Iterable[str] = ()) -> list[str]:
    """Detect regex features from metadata text and sample values."""

    combined = " ".join([text, *[str(value) for value in samples if value is not None]])
    return [rule.name for rule in REGEX_RULES if rule.pattern.search(combined)]


def evidence_from_regex_features(features: Iterable[str]) -> list[str]:
    """Convert regex feature names to human-readable evidence labels."""

    feature_set = set(features)
    return [rule.evidence for rule in REGEX_RULES if rule.name in feature_set]


def regex_weight_tokens(features: Iterable[str]) -> list[str]:
    """Return repeated feature tokens so strong regex signals affect Naive Bayes."""

    feature_set = set(features)
    tokens: list[str] = []
    for rule in REGEX_RULES:
        if rule.name in feature_set:
            tokens.extend([f"regex_{rule.name}"] * rule.weight)
    return tokens


def _has_chinese(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _bucket_length(avg_length: float) -> str:
    if avg_length == 0:
        return "len_empty"
    if avg_length <= 4:
        return "len_short"
    if avg_length <= 12:
        return "len_medium"
    if avg_length <= 32:
        return "len_long"
    return "len_very_long"


def build_sample_profile(samples: Iterable[str]) -> dict[str, object]:
    """Convert raw samples into safe statistical and format features.

    Raw sample values are never returned.
    """

    values = [str(value) for value in samples if value is not None and str(value) != ""]
    lengths = [len(value) for value in values]
    sample_count = len(values)
    unique_count = len(set(values))
    regex_ratios: dict[str, float] = {}

    for rule in REGEX_RULES:
        if sample_count == 0:
            ratio = 0.0
        else:
            ratio = sum(1 for value in values if rule.pattern.search(value)) / sample_count
        if ratio > 0:
            regex_ratios[rule.name] = round(ratio, 4)

    avg_length = round(mean(lengths), 2) if lengths else 0.0
    fixed_length = len(set(lengths)) == 1 if lengths else False
    numeric_ratio = round(sum(1 for value in values if value.isdigit()) / sample_count, 4) if sample_count else 0.0
    chinese_ratio = round(sum(1 for value in values if _has_chinese(value)) / sample_count, 4) if sample_count else 0.0
    unique_ratio = round(unique_count / sample_count, 4) if sample_count else 0.0

    return {
        "sample_count": sample_count,
        "avg_length": avg_length,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "fixed_length": fixed_length,
        "length_bucket": _bucket_length(avg_length),
        "numeric_ratio": numeric_ratio,
        "contains_chinese_ratio": chinese_ratio,
        "unique_ratio": unique_ratio,
        "high_unique_ratio": unique_ratio >= 0.8 if sample_count >= 3 else False,
        "regex_match_ratios": regex_ratios,
    }


def sample_profile_tokens(profile: dict[str, object]) -> list[str]:
    """Convert a sample profile into model tokens."""

    tokens = [
        f"sample_count_{profile.get('sample_count', 0)}",
        str(profile.get("length_bucket", "len_unknown")),
    ]
    if profile.get("fixed_length"):
        tokens.append("sample_fixed_length")
    if float(profile.get("numeric_ratio", 0.0)) >= 0.8:
        tokens.append("sample_mostly_numeric")
    if float(profile.get("contains_chinese_ratio", 0.0)) >= 0.5:
        tokens.append("sample_contains_chinese")
    if profile.get("high_unique_ratio"):
        tokens.append("sample_high_unique_ratio")

    regex_ratios = profile.get("regex_match_ratios", {})
    if isinstance(regex_ratios, dict):
        for name, ratio in regex_ratios.items():
            if float(ratio) >= 0.6:
                tokens.extend([f"sample_regex_{name}"] * 3)
            elif float(ratio) > 0:
                tokens.append(f"sample_regex_{name}")
    return tokens

