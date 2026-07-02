from data_classification_tool.bayes_classifier import DataFieldBayesClassifier
from data_classification_tool.synthetic_data import (
    generate_synthetic_training_records,
    synthetic_label_set,
)
from data_classification_tool.synthetic_training import train_synthetic_until_target


def test_generate_synthetic_training_records_is_deterministic() -> None:
    records_a = generate_synthetic_training_records(records_per_category=3, seed=7)
    records_b = generate_synthetic_training_records(records_per_category=3, seed=7)

    assert records_a == records_b
    assert len(records_a) == len(synthetic_label_set()) * 3
    assert {record["label"] for record in records_a} == synthetic_label_set()
    assert all(record["sample_values"] for record in records_a)


def test_synthetic_records_can_train_bayes_classifier() -> None:
    records = generate_synthetic_training_records(records_per_category=8, seed=11)
    classifier = DataFieldBayesClassifier().fit(records)

    id_prediction = classifier.predict(
        {
            "table_name": "user_auth",
            "table_comment": "实名认证表",
            "column_name": "id_card_no",
            "column_comment": "身份证号码",
            "data_type": "varchar",
            "sample_values": ["110101199001011234"],
        }
    )
    phone_prediction = classifier.predict(
        {
            "table_name": "user_contact",
            "table_comment": "用户联系方式表",
            "column_name": "mobile_phone",
            "column_comment": "手机号",
            "data_type": "varchar",
            "sample_values": ["13800001111"],
        }
    )

    assert id_prediction.predicted_category == "个人身份信息"
    assert phone_prediction.predicted_category == "个人联系方式"


def test_train_synthetic_until_target_stops_and_saves_model(tmp_path) -> None:
    model_path = tmp_path / "synthetic_until_target.joblib"

    result = train_synthetic_until_target(
        model_path=model_path,
        target_accuracy=0.7,
        max_rounds=3,
        initial_records_per_category=4,
        records_increment=2,
        validation_records_per_category=2,
        validation_seed_count=2,
        test_records_per_category=2,
    )

    assert model_path.exists()
    assert result.history
    assert result.best_round <= 3
    assert result.best_validation_accuracy >= 0.7
    assert 0 <= result.final_test_accuracy <= 1
