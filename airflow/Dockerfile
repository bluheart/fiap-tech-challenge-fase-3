# Use Airflow 2.7.3 with Python 3.11 (the latest supported version)
FROM apache/airflow:2.7.3-python3.11

USER root

# Install system-level PostgreSQL dev libraries (needed for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies including the PostgreSQL driver
RUN pip install --no-cache-dir \
    "datasets>=2.16.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "numpy>=1.24.0" \
    "pyarrow>=14.0.0" \
    "psycopg2-binary"