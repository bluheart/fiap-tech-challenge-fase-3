FROM apache/airflow:2.7.3  # ← No LocalExecutor parallelism bug

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "numpy>=1.24.0" \
    "pyarrow>=14.0.0"