from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.feature_extractor import FieldFeatureExtractor
from data_classification_tool.models import FieldRecord


def test_feature_extractor_does_not_leak_raw_samples() -> None:
    record = FieldRecord(
        table_name="user_auth",
        table_comment="用户实名认证表",
        column_name="id_card",
        column_comment="身份证号",
        data_type="varchar",
        sample_values=("110101199001011234",),
        is_unique=True,
    )

    features = FieldFeatureExtractor().extract(record)

    assert "110101199001011234" not in features.feature_text
    assert "regex_id_number" in features.feature_text
    assert "身份证正则匹配" in features.evidence_features


def test_data_field_bayes_classifier_predicts_with_explanations() -> None:
    train_records = [
        {
            "table_name": "user_auth",
            "table_comment": "用户实名认证表",
            "column_name": "id_card_no",
            "column_comment": "身份证号码",
            "data_type": "varchar",
            "sample_values": ["110101********1234"],
            "label": "个人身份信息",
        },
        {
            "table_name": "contact_book",
            "table_comment": "联系人表",
            "column_name": "mobile_phone",
            "column_comment": "手机号",
            "data_type": "varchar",
            "sample_values": ["138****0000"],
            "label": "个人联系方式",
        },
        {
            "table_name": "access_log",
            "table_comment": "访问日志表",
            "column_name": "client_ip",
            "column_comment": "客户端IP",
            "data_type": "varchar",
            "sample_values": ["192.168.1.1"],
            "label": "系统运行日志",
        },
    ]
    classifier = DataFieldBayesClassifier().fit(train_records)

    prediction = classifier.predict(
        {
            "table_name": "user_auth",
            "table_comment": "用户认证表",
            "column_name": "cert_no",
            "column_comment": "身份证号",
            "data_type": "varchar",
            "sample_values": ["320102********5678"],
        }
    )
    llm_payload = classifier.to_llm_grading_input(
        {
            "table_name": "user_auth",
            "table_comment": "用户认证表",
            "column_name": "cert_no",
            "column_comment": "身份证号",
            "data_type": "varchar",
            "sample_values": ["320102********5678"],
        },
        prediction,
    )

    assert prediction.predicted_category == "个人身份信息"
    assert prediction.top_candidates
    assert "身份证正则匹配" in prediction.evidence_features
    assert llm_payload["bayes_category"] == "个人身份信息"
    assert "sample_profile" in llm_payload

