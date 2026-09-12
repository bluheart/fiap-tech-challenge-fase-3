FROM apache/airflow:2.7.3-python3.11

USER root

# Trocar para mirrors brasileiros antes do apt-get
RUN sed -i 's|deb.debian.org|mirror.usp.br|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org|mirror.usp.br|g' /etc/apt/sources.list.d/debian.sources

# Agora instalar as dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    "datasets>=2.16.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "numpy>=1.24.0" \
    "pyarrow>=14.0.0" \
    "psycopg2-binary"