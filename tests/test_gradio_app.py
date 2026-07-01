from data_classification_tool.gradio_app import build_app


def test_gradio_app_builds() -> None:
    app = build_app()

    assert app is not None

