from pathlib import Path

from data_classification_tool.catalog import load_label_catalog
from data_classification_tool.grader import RuleBasedGrader
from data_classification_tool.models import ClassificationResult, FieldRecord
from data_classification_tool.rule_engine import RiskRuleEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RISK_RULE_CSV = PROJECT_ROOT / "data" / "risk_rule_catalog.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "field_label_catalog.csv"


def _classification(label_id: str, category: str, patterns: list[str] | None = None) -> ClassificationResult:
    return ClassificationResult(
        label_id=label_id,
        category=category,
        confidence=0.95,
        matched_patterns=patterns or [],
    )


def test_risk_rule_engine_matches_sensitive_identity() -> None:
    engine = RiskRuleEngine.from_csv(RISK_RULE_CSV)
    record = FieldRecord(
        table_name="user_auth",
        column_name="id_card_no",
        column_comment="身份证号码",
        data_type="varchar",
    )

    matches = engine.match(record, _classification("id_number", "身份证件号", ["looks_like_id_number"]))

    assert matches
    assert matches[0].rule.rule_id == "PI-SPI-001"
    assert matches[0].rule.minimum_level == "L4"
    assert "sensitive_personal_info" in matches[0].rule.risk_tags


def test_rule_based_grader_keeps_secret_at_l4() -> None:
    catalog = load_label_catalog(CATALOG_CSV)
    grader = RuleBasedGrader(catalog, risk_rule_csv=RISK_RULE_CSV)
    record = FieldRecord(
        table_name="api_client",
        column_name="client_secret",
        column_comment="第三方应用密钥",
        data_type="varchar",
        sample_value="sec_***",
    )

    result = grader.grade(record, _classification("auth_token", "密钥令牌", ["looks_like_secret"]))

    assert result.level == "L4"
    assert "SYS-SECRET-001" in result.matched_rules
    assert "credential_or_secret" in result.risk_tags


def test_rule_based_grader_matches_carbon_data() -> None:
    catalog = load_label_catalog(CATALOG_CSV)
    grader = RuleBasedGrader(catalog, risk_rule_csv=RISK_RULE_CSV)
    record = FieldRecord(
        table_name="carbon_report",
        column_name="carbon_emission_amount",
        column_comment="企业碳排放量",
        data_type="decimal",
        sample_value="10240.50",
    )

    result = grader.grade(record, _classification("unknown", "其他"))

    assert result.level == "L3"
    assert "CARBON-EMISSION-001" in result.matched_rules
    assert "carbon_data_risk" in result.risk_tags

