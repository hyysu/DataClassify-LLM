from importlib.metadata import PackageNotFoundError, version

PACKAGES = [
    "pandas",
    "scikit-learn",
    "openpyxl",
    "joblib",
    "pydantic",
    "requests",
    "python-dotenv",
    "openai",
    "gradio",
    "jieba",
]

print("Python dependency smoke test")
for package in PACKAGES:
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
