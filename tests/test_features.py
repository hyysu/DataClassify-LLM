from data_classification_tool.features import build_feature_text, detect_patterns, split_identifier
from data_classification_tool.models import FieldRecord


def test_split_identifier_handles_snake_case() -> None:
    assert split_identifier("receiver_mobile_phone") == ["receiver", "mobile", "phone"]


def test_detect_patterns_for_mobile_phone() -> None:
    record = FieldRecord(
        table_name="user",
        column_name="phone",
        column_comment="手机号",
        data_type="varchar",
        sample_value="138****0000",
        business_context="用户管理",
    )

    assert "mobile_phone" in detect_patterns(record)


def test_build_feature_text_contains_pattern_markers() -> None:
    record = FieldRecord(
        table_name="api_client",
        column_name="client_secret",
        column_comment="客户端密钥",
        data_type="varchar",
        sample_value="sec_***",
        business_context="开放平台",
    )

    text = build_feature_text(record)

    assert "client_secret" in text
    assert "regex_secret_token" in text
