# 数据分类分级工具

本项目用于构建字段级数据分类分级原型：

- 使用贝叶斯模型对数据字段进行分类打标；
- 使用通用大模型 API 对字段进行安全分级、理由生成和保护措施建议；
- 支持后续扩展为 CSV/Excel/数据库表结构扫描工具。

## 环境

- Conda 安装位置：`D:\Miniconda3`
- Conda 环境目录：`D:\conda\envs`
- Conda 包缓存目录：`D:\conda\pkgs`
- 本项目环境名：`data-classification`
- Python 版本：3.11

## 验证环境

```powershell
& "D:\Miniconda3\Scripts\conda.exe" run -n data-classification python D:\classification\data-classification-tool\scripts\smoke_test.py
```

## 运行原型

训练贝叶斯字段分类模型：

```powershell
& "D:\conda\envs\data-classification\python.exe" D:\classification\data-classification-tool\scripts\train_model.py
```

分析示例字段并导出报告：

```powershell
& "D:\conda\envs\data-classification\python.exe" D:\classification\data-classification-tool\scripts\analyze_demo.py
```

评估贝叶斯分类效果：

```powershell
& "D:\conda\envs\data-classification\python.exe" D:\classification\data-classification-tool\scripts\evaluate_model.py
```

启动本地页面：

```powershell
& "D:\conda\envs\data-classification\python.exe" D:\classification\data-classification-tool\scripts\run_gradio.py
```

浏览器访问：

```text
http://127.0.0.1:7860
```

输出文件：

- `D:\classification\data-classification-tool\reports\demo_report.csv`
- `D:\classification\data-classification-tool\reports\demo_report.json`
- `D:\classification\data-classification-tool\reports\evaluation\evaluation_metrics.json`
- `D:\classification\data-classification-tool\reports\evaluation\evaluation_predictions.csv`
- `D:\classification\data-classification-tool\reports\evaluation\confusion_matrix.csv`
- `D:\classification\data-classification-tool\reports\evaluation\evaluation_report.xlsx`
- 页面生成的报告位于：`D:\classification\data-classification-tool\reports\ui`

运行测试：

```powershell
& "D:\conda\envs\data-classification\python.exe" -m pytest D:\classification\data-classification-tool\tests
```

当前默认使用本地规则分级，不需要大模型 API key。后续有 key 后，再切换到 `--grader llm`。

## 字段分类模块

新的字段级贝叶斯分类模块支持 JSON/CSV 训练数据、字段特征抽取、top-k 候选分类、低置信度复核标记和 LLM 分级输入转换。

训练字段分类模型：

```powershell
& "D:\conda\envs\data-classification\python.exe" D:\classification\data-classification-tool\scripts\train.py
```

预测字段分类：

```powershell
& "D:\conda\envs\data-classification\python.exe" D:\classification\data-classification-tool\scripts\predict.py
```

评估字段分类模型：

```powershell
& "D:\conda\envs\data-classification\python.exe" D:\classification\data-classification-tool\scripts\evaluate.py
```

详细说明见：

`D:\classification\data-classification-tool\docs\贝叶斯字段分类模块说明.md`

## 激活环境

如果希望在当前 PowerShell 中激活环境：

```powershell
(& "D:\Miniconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate data-classification
cd D:\classification\data-classification-tool
```

或使用项目辅助脚本：

```powershell
. D:\classification\data-classification-tool\scripts\activate_project.ps1
```

## 目录说明

- `data/`：样本数据、字段样本集、标签体系表
- `docs/`：方案、调研、环境说明
- `src/`：后续 Python 源码
- `scripts/`：环境验证、训练、推理等脚本
- `environment.yml`：conda 环境声明
- `.env.example`：大模型 API 配置模板
