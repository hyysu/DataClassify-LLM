"""Shared paths and classifier defaults."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_BAYES_MODEL_PATH = MODEL_DIR / "data_field_bayes.joblib"
DEFAULT_FIELD_TRAINING_JSON = DATA_DIR / "sample_train_data.json"
DEFAULT_FIELD_PREDICT_JSON = DATA_DIR / "sample_predict_data.json"
DEFAULT_RISK_RULE_CSV = DATA_DIR / "risk_rule_catalog.csv"
DEFAULT_REVIEW_THRESHOLD = 0.6
DEFAULT_TOP_K = 3
