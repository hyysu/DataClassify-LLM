"""Gradio UI for the data classification and grading prototype."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data_classification_tool.cli import DEFAULT_CATALOG, DEFAULT_DEMO, DEFAULT_MODEL, DEFAULT_TRAINING, PROJECT_ROOT
from data_classification_tool.io import flatten_result
from data_classification_tool.pipeline import analyze_fields, train_classifier


UI_REPORT_DIR = PROJECT_ROOT / "reports" / "ui"
UI_TEMP_DIR = PROJECT_ROOT / "reports" / "gradio_tmp"

os.environ.setdefault("GRADIO_TEMP_DIR", str(UI_TEMP_DIR))
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

APP_THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
APP_CSS = """
.gradio-container { max-width: 1440px !important; }
.panel { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
"""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _file_path(file_value: Any) -> Path | None:
    if file_value is None:
        return None
    if isinstance(file_value, (str, Path)):
        return Path(file_value)
    if hasattr(file_value, "name"):
        return Path(file_value.name)
    if isinstance(file_value, dict) and file_value.get("path"):
        return Path(file_value["path"])
    return None


def _ensure_model(retrain: bool) -> None:
    if retrain or not DEFAULT_MODEL.exists():
        train_classifier(DEFAULT_TRAINING, DEFAULT_MODEL)


def _write_excel_report(frame: pd.DataFrame, output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="classification_report")


def _analyze(input_csv: Path, retrain: bool, grader_name: str) -> tuple[str, pd.DataFrame, str, str, str]:
    if not input_csv.exists():
        raise gr.Error(f"输入文件不存在：{input_csv}")

    _ensure_model(retrain)
    UI_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    UI_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    stem = input_csv.stem or "fields"
    run_id = f"{stem}_{_timestamp()}"
    output_csv = UI_REPORT_DIR / f"{run_id}_report.csv"
    output_json = UI_REPORT_DIR / f"{run_id}_report.json"
    output_xlsx = UI_REPORT_DIR / f"{run_id}_report.xlsx"

    results = analyze_fields(
        input_csv=input_csv,
        catalog_csv=DEFAULT_CATALOG,
        model_path=DEFAULT_MODEL,
        output_csv=output_csv,
        output_json=output_json,
        grader_name=grader_name,
    )
    frame = pd.DataFrame([flatten_result(result) for result in results])
    _write_excel_report(frame, output_xlsx)

    total = len(frame)
    review_count = int(frame["requires_review"].sum()) if total else 0
    high_count = int(frame["level"].isin(["L3", "L4"]).sum()) if total else 0
    level_counts = frame["level"].value_counts().to_dict() if total else {}
    summary = (
        f"字段数：{total}\n\n"
        f"敏感及高敏字段：{high_count}\n\n"
        f"需要复核：{review_count}\n\n"
        f"等级分布：{level_counts}"
    )

    display_columns = [
        "table_name",
        "column_name",
        "column_comment",
        "label_id",
        "category",
        "classification_confidence",
        "level",
        "level_name",
        "requires_review",
        "controls",
        "reason",
    ]
    return summary, frame[display_columns], str(output_csv), str(output_json), str(output_xlsx)


def analyze_uploaded_file(file_value: Any, retrain: bool, grader_name: str) -> tuple[str, pd.DataFrame, str, str, str]:
    input_csv = _file_path(file_value)
    if input_csv is None:
        raise gr.Error("请上传字段 CSV 文件，或使用示例数据。")
    return _analyze(input_csv, retrain, grader_name)


def analyze_demo_file(retrain: bool, grader_name: str) -> tuple[str, pd.DataFrame, str, str, str]:
    return _analyze(DEFAULT_DEMO, retrain, grader_name)


def load_demo_preview() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DEMO).fillna("")


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks application."""

    with gr.Blocks(title="数据分类分级工具") as app:
        gr.Markdown("# 数据分类分级工具")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["panel"]):
                file_input = gr.File(
                    label="字段CSV",
                    file_count="single",
                    file_types=[".csv"],
                    type="filepath",
                )
                retrain = gr.Checkbox(label="运行前重新训练模型", value=False)
                grader_name = gr.Radio(
                    label="分级方式",
                    choices=["rule", "llm"],
                    value="rule",
                )
                with gr.Row():
                    analyze_upload_button = gr.Button("分析上传文件", variant="primary")
                    analyze_demo_button = gr.Button("分析示例数据")
                sample_file = gr.File(label="示例CSV", value=str(DEFAULT_DEMO), interactive=False)

            with gr.Column(scale=2, elem_classes=["panel"]):
                summary = gr.Markdown("等待分析")
                result_table = gr.Dataframe(
                    label="分类分级结果",
                    interactive=False,
                    wrap=True,
                    max_height=520,
                )

        with gr.Row(equal_height=True):
            csv_output = gr.File(label="CSV报告")
            json_output = gr.File(label="JSON报告")
            xlsx_output = gr.File(label="Excel报告")

        with gr.Accordion("示例数据预览", open=False):
            demo_preview = gr.Dataframe(value=load_demo_preview, interactive=False, wrap=True)

        analyze_upload_button.click(
            fn=analyze_uploaded_file,
            inputs=[file_input, retrain, grader_name],
            outputs=[summary, result_table, csv_output, json_output, xlsx_output],
        )
        analyze_demo_button.click(
            fn=analyze_demo_file,
            inputs=[retrain, grader_name],
            outputs=[summary, result_table, csv_output, json_output, xlsx_output],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        theme=APP_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
