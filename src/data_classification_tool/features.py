"""Backward-compatible feature helpers.

New code should use :mod:`data_classification_tool.feature_extractor`.
"""

from __future__ import annotations

from data_classification_tool.feature_extractor import (
    DEFAULT_EXTRACTOR,
    FieldFeatureExtractor,
    build_feature_text,
    detect_patterns,
    split_identifier,
    tokenize_chinese,
)

__all__ = [
    "DEFAULT_EXTRACTOR",
    "FieldFeatureExtractor",
    "build_feature_text",
    "detect_patterns",
    "split_identifier",
    "tokenize_chinese",
]

