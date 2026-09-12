# Usar uma versão recente do Airflow que já corrigiu o bug do LocalExecutor
FROM apache/airflow:3.4.0

USER root

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Instalar dependências Python necessárias para suas DAGs
# Nota: datasets >=5.0.1, pandas >=3.0.5, scikit-learn >=1.9.0, numpy >=2.5.2
# Estes são necessários para a lógica do seu pipeline de treinamento.
RUN pip install --no-cache-dir \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "numpy>=1.24.0" \
    "pyarrow>=14.0.0"