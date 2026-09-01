import logging
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def alert_failure(context):
    """Callback chamado quando task falha (substitua por Slack/PagerDuty em prod)."""
    ti = context["task_instance"]
    logger.error(
        "🚨 ALERTA: Task %s falhou na DAG %s (run %s). Log: %s",
        ti.task_id, ti.dag_id, ti.run_id, ti.log_url,
    )

default_args = {
    'owner': 'caio',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": alert_failure,
}

DATA_DIR = Path("/tmp/ml_pipeline")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data(**context):
    """Carrega dados de treino do CSV"""
    df = pd.read_csv('../../data/laudos_treinamento.csv')
    context['ti'].xcom_push(key='data_shape', value=df.shape)
    path = DATA_DIR / "raw_data.csv"
    df.to_csv(path, index=False)
    return path


def preprocess(**context):
    """Pré-processamento dos textos"""
    data_path = context['ti'].xcom_pull(task_ids='load_data')
    df = pd.read_csv(data_path)
    
    # Limpeza simples
    df['texto_limpo'] = df['texto'].str.lower().str.replace(r'[^a-z ]', '')
    
    path = DATA_DIR / 'processed_data.csv'
    df.to_csv(path, index=False)
    return path


def train_model(**context):
    """Treina modelo TF-IDF + Random Forest"""
    data_path = context['ti'].xcom_pull(task_ids='preprocess')
    df = pd.read_csv(data_path)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['texto_limpo'])
    y = df['classificacao']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    vec_path = DATA_DIR / 'vectorizer.pkl'
    model_path = DATA_DIR / 'model.pkl'

    # Salva artefatos
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)
    
    return "Modelo treinado com sucesso"


def validate_model(**context):
    """Validação básica do modelo treinado"""
    model_path = DATA_DIR / 'model.pkl'
    model = joblib.load(model_path)
    assert hasattr(model, 'predict_proba'), "Modelo não possui predict_proba"
    return "Validação OK"


with DAG(
    dag_id='model_training_pipeline',
    default_args=default_args,
    description='Pipeline de treinamento do classificador de laudos',
    schedule='@weekly',
    catchup=False,
    tags=['ml', 'training'],
) as dag:
    
    t1 = PythonOperator(task_id='load_data', python_callable=load_data)
    
    t2 = PythonOperator(task_id='preprocess', python_callable=preprocess)
    
    t3 = PythonOperator(task_id='train_model', python_callable=train_model)
    
    t4 = PythonOperator(task_id='validate_model', python_callable=validate_model)

    t1 >> t2 >> t3 >> t4 # type: ignore