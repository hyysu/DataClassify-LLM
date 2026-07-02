from __future__ import annotations

from pathlib import Path
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_classification_tool.cli import DEFAULT_CATALOG, DEFAULT_MODEL, DEFAULT_TRAINING
from data_classification_tool.io import flatten_result
from data_classification_tool.pipeline import analyze_fields, train_classifier


INPUT_CSV = PROJECT_ROOT / "data" / "gradio_generated_fields.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


TITLE = load_font(38, bold=True)
H2 = load_font(25, bold=True)
BODY = load_font(20)
SMALL = load_font(17)
TABLE = load_font(16)
TABLE_BOLD = load_font(16, bold=True)


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, value: str, accent: str) -> None:
    draw.rounded_rectangle(box, radius=14, fill="white", outline="#d8dee8", width=2)
    x1, y1, _, _ = box
    draw.rectangle((x1, y1, x1 + 8, box[3]), fill=accent)
    draw.text((x1 + 26, y1 + 22), title, font=BODY, fill="#465064")
    draw.text((x1 + 26, y1 + 58), value, font=TITLE, fill="#111827")


def create_summary_image(rows: list[dict[str, object]]) -> None:
    width, height = 1500, 840
    image = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(image)

    total = len(rows)
    high_count = sum(1 for row in rows if row["level"] in {"L3", "L4"})
    review_count = sum(1 for row in rows if row["requires_review"])
    level_counts = {level: sum(1 for row in rows if row["level"] == level) for level in ["L1", "L2", "L3", "L4"]}

    draw.text((56, 44), "DataClassify-LLM 数据分类分级演示", font=TITLE, fill="#111827")
    draw.text((58, 100), "基于现有贝叶斯字段分类模型与本地规则分级，对测试字段集进行自动识别。", font=BODY, fill="#4b5563")

    card_y = 160
    card_w, card_h, gap = 320, 150, 28
    draw_card(draw, (56, card_y, 56 + card_w, card_y + card_h), "字段数量", str(total), "#2563eb")
    draw_card(draw, (56 + (card_w + gap), card_y, 56 + 2 * card_w + gap, card_y + card_h), "敏感及高敏字段", str(high_count), "#dc2626")
    draw_card(draw, (56 + 2 * (card_w + gap), card_y, 56 + 3 * card_w + 2 * gap, card_y + card_h), "需要人工复核", str(review_count), "#f59e0b")
    draw_card(draw, (56 + 3 * (card_w + gap), card_y, 56 + 4 * card_w + 3 * gap, card_y + card_h), "数据等级", "L1-L4", "#059669")

    draw.text((58, 322), "测试字段集覆盖身份、联系方式、财产、健康、日志、交易、经营与公开数据等典型场景。", font=SMALL, fill="#6b7280")

    panel = (56, 360, width - 56, 780)
    draw.rounded_rectangle(panel, radius=16, fill="white", outline="#d8dee8", width=2)
    draw.text((90, 392), "等级分布", font=H2, fill="#111827")

    colors = {"L1": "#22c55e", "L2": "#3b82f6", "L3": "#f59e0b", "L4": "#ef4444"}
    max_count = max(level_counts.values()) or 1
    start_x, start_y = 120, 470
    bar_h, bar_gap = 54, 34
    for index, level in enumerate(["L1", "L2", "L3", "L4"]):
        y = start_y + index * (bar_h + bar_gap)
        count = level_counts[level]
        bar_w = int(900 * count / max_count)
        draw.text((90, y + 12), level, font=BODY, fill="#111827")
        draw.rounded_rectangle((150, y, 150 + 900, y + bar_h), radius=10, fill="#eef2f7")
        draw.rounded_rectangle((150, y, 150 + bar_w, y + bar_h), radius=10, fill=colors[level])
        draw.text((1080, y + 12), f"{count} 个字段", font=BODY, fill="#111827")

    image.save(OUTPUT_DIR / "demo-summary.png", quality=95)


def short_text(value: object, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def create_table_image(rows: list[dict[str, object]]) -> None:
    width, height = 1800, 1120
    image = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(image)

    draw.text((56, 36), "字段分类分级结果预览", font=TITLE, fill="#111827")
    draw.text((58, 90), "README 截图展示前 14 条测试字段结果，完整结果可通过 Gradio 页面或报告文件查看。", font=BODY, fill="#4b5563")

    x0, y0 = 56, 150
    columns = [
        ("表名", 190),
        ("字段名", 210),
        ("字段注释", 220),
        ("分类", 170),
        ("置信度", 110),
        ("等级", 90),
        ("复核", 80),
        ("判断理由", 660),
    ]
    row_h = 62
    header_h = 54
    draw.rounded_rectangle((x0, y0, width - 56, y0 + header_h + row_h * 14), radius=12, fill="white", outline="#d8dee8", width=2)
    draw.rectangle((x0, y0, width - 56, y0 + header_h), fill="#e8eefc")

    x = x0
    for name, col_w in columns:
        draw.text((x + 12, y0 + 16), name, font=TABLE_BOLD, fill="#111827")
        x += col_w

    level_colors = {"L1": "#16a34a", "L2": "#2563eb", "L3": "#d97706", "L4": "#dc2626"}
    for row_index, row in enumerate(rows[:14]):
        y = y0 + header_h + row_index * row_h
        if row_index % 2:
            draw.rectangle((x0, y, width - 56, y + row_h), fill="#fbfcff")
        x = x0
        values = [
            short_text(row["table_name"], 18),
            short_text(row["column_name"], 20),
            short_text(row["column_comment"], 18),
            short_text(row["category"], 12),
            f"{float(row['classification_confidence']):.4f}",
            str(row["level"]),
            "是" if row["requires_review"] else "否",
            short_text(row["reason"], 48),
        ]
        for col_index, ((_, col_w), value) in enumerate(zip(columns, values)):
            fill = level_colors.get(str(value), "#1f2937") if col_index == 5 else "#1f2937"
            if col_index == 7:
                lines = textwrap.wrap(str(value), width=34)
                for line_index, line in enumerate(lines[:2]):
                    draw.text((x + 12, y + 10 + line_index * 22), line, font=TABLE, fill=fill)
            else:
                draw.text((x + 12, y + 20), str(value), font=TABLE, fill=fill)
            x += col_w
        draw.line((x0, y + row_h, width - 56, y + row_h), fill="#e5e7eb")

    image.save(OUTPUT_DIR / "demo-results-table.png", quality=95)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_MODEL.exists():
        train_classifier(DEFAULT_TRAINING, DEFAULT_MODEL)
    results = analyze_fields(
        input_csv=INPUT_CSV,
        catalog_csv=DEFAULT_CATALOG,
        model_path=DEFAULT_MODEL,
        grader_name="rule",
    )
    rows = [flatten_result(result) for result in results]
    create_summary_image(rows)
    create_table_image(rows)
    print(f"Generated README images in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
