from setuptools import setup, find_packages

setup(
    name="retail-brand-sentiment-monitor",
    version="1.0.0",
    description="Brand monitoring and sentiment analysis system for DTC retail",
    author="NLP Team",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "torch>=2.1.0",
        "bertopic>=0.16.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "pydantic>=2.5.0",
        "emoji>=2.8.0",
        "scikit-learn>=1.3.0",
    ],
)
