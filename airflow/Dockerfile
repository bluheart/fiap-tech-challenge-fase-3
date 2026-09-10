FROM apache/airflow:3.3.1

# Switch to root to install system dependencies
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install Python dependencies needed for the DAGs
RUN pip install --no-cache-dir \
    "datasets>=5.0.1" \
    "pandas>=3.0.5" \
    "scikit-learn>=1.9.0" \
    "numpy>=2.5.2"