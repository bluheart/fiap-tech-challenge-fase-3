import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from airflow import DAG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos compartilhados (volumes Docker)
SHARED_DATA_PATH = '/shared/data'
SHARED_MODELS_PATH = '/shared/models'
SHARED_VECTORIZERS_PATH = '/shared/vectorizers'
LOCAL_DATA_PATH = '/opt/airflow/data/laudos_treinamento.parquet'

# Garantir que os diretórios existam
for path in [SHARED_DATA_PATH, SHARED_MODELS_PATH, SHARED_VECTORIZERS_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)

default_args = {
    'owner': 'medical_ai_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 9, 1, tzinfo=ZoneInfo("America/Sao_Paulo")),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def load_data(**context):
    """Carregar dados médicos para treinamento do Parquet compartilhado"""
    try:
        # Tentar carregar do volume compartilhado primeiro
        if os.path.exists(SHARED_DATA_PATH + '/laudos_treinamento.parquet'):
            data_path = SHARED_DATA_PATH + '/laudos_treinamento.parquet'
            logger.info(f"Carregando dados do volume compartilhado: {data_path}")
        elif os.path.exists(LOCAL_DATA_PATH):
            data_path = LOCAL_DATA_PATH
            logger.info(f"Carregando dados do volume local do Airflow: {data_path}")
        else:
            raise FileNotFoundError("Arquivo de dados não encontrado em nenhum local")
        
        # Carregar dados
        df = pd.read_parquet(data_path)
        logger.info(f"Dados carregados: {len(df)} amostras")
        logger.info(f"Colunas: {df.columns.tolist()}")
        
        # Verificar colunas necessárias
        required_columns = ['texto', 'target']
        for col in required_columns:
            if col not in df.columns:
                # Tentar mapear colunas comuns
                if col == 'texto' and 'text' in df.columns:
                    df['texto'] = df['text']
                elif col == 'texto' and 'laudo' in df.columns:
                    df['texto'] = df['laudo']
                elif col == 'texto' and 'descricao' in df.columns:
                    df['texto'] = df['descricao']
                else:
                    raise ValueError(f"Coluna '{col}' não encontrada no dataset")
        
        # Salvar cópia no volume compartilhado para a API
        df.to_parquet(SHARED_DATA_PATH + '/laudos_treinamento.parquet', index=False)
        logger.info(f"Dados salvos no volume compartilhado: {SHARED_DATA_PATH}")
        
        # Salvar informações no contexto
        context['task_instance'].xcom_push(key='data_path', value=data_path)
        context['task_instance'].xcom_push(key='data_shape', value=str(df.shape))
        context['task_instance'].xcom_push(key='columns', value=json.dumps(df.columns.tolist()))
        
        # Salvar DataFrame como pickle para uso nas próximas tasks
        df.to_pickle('/tmp/df_loaded.pkl')
        
        return f"Data loaded successfully: {len(df)} samples, columns: {df.columns.tolist()}"
    
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e!s}")
        raise

def preprocess_data(**context):
    """Pré-processamento dos dados"""
    try:
        # Carregar DataFrame do pickle
        df = pd.read_pickle('/tmp/df_loaded.pkl')
        
        logger.info(f"Pré-processando {len(df)} amostras")
        
        # Selecionar colunas relevantes
        text_col = 'texto'
        target_col = 'classificacao'
        
        logger.info(f"Coluna de texto: {text_col}, Coluna alvo: {target_col}")
        
        # Limpeza básica
        df[text_col] = df[text_col].astype(str).str.lower().str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        df = df.dropna(subset=[text_col, target_col])
        
        # Normalizar classes
        class_mapping = {
            'normal': 0,
            'urgente': 1,
        }
        
        # Aplicar mapeamento se necessário
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].map(class_mapping).fillna(df[target_col])
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_col], 
            df[target_col], 
            test_size=0.2, 
            random_state=42,
            stratify=df[target_col] if len(df[target_col].unique()) > 1 else None
        )
        
        logger.info(f"Split realizado: Train={len(X_train)}, Test={len(X_test)}")
        
        # Salvar splits para uso posterior
        X_train.to_pickle('/tmp/X_train.pkl')
        X_test.to_pickle('/tmp/X_test.pkl')
        y_train.to_pickle('/tmp/y_train.pkl')
        y_test.to_pickle('/tmp/y_test.pkl')
        
        # Salvar no contexto
        context['task_instance'].xcom_push(key='train_size', value=str(len(X_train)))
        context['task_instance'].xcom_push(key='test_size', value=str(len(X_test)))
        context['task_instance'].xcom_push(key='classes', value=json.dumps(df[target_col].unique().tolist()))
        
        return f"Preprocessing completed: {len(X_train)} train samples, {len(X_test)} test samples"
    
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e!s}")
        raise

def train_model(**context):
    """Treinar modelo de classificação e salvar no volume compartilhado"""
    try:
        # Carregar dados pré-processados
        X_train = pd.read_pickle('/tmp/X_train.pkl')
        y_train = pd.read_pickle('/tmp/y_train.pkl')
        
        logger.info(f"Treinando modelo com {len(X_train)} amostras")
        
        # Vectorizar texto
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_vectorized = vectorizer.fit_transform(X_train)
        
        # Treinar modelo
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train_vectorized, y_train)
        
        # Criar timestamp para versionamento
        timestamp = datetime.now(tz=ZoneInfo("America/Sao_Paulo")).strftime('%Y%m%d_%H%M%S')
        model_version = f"v_{timestamp}"
        
        # Preparar dados do modelo
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'version': model_version,
            'timestamp': datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            'feature_names': vectorizer.get_feature_names_out().tolist(),
            'classes': model.classes_.tolist(),
            'n_features': X_train_vectorized.shape[1]
        }
        
        # Salvar modelo no volume compartilhado
        shared_model_path = f"{SHARED_MODELS_PATH}/{model_version}_model.pkl"
        with open(shared_model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Salvar vectorizer separadamente
        shared_vectorizer_path = f"{SHARED_VECTORIZERS_PATH}/{model_version}_vectorizer.pkl"
        with open(shared_vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Também salvar como "latest" para a API
        latest_model_path = f"{SHARED_MODELS_PATH}/latest_model.pkl"
        with open(latest_model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        latest_vectorizer_path = f"{SHARED_VECTORIZERS_PATH}/latest_vectorizer.pkl"
        with open(latest_vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Salvar metadados do modelo
        metadata = {
            'version': model_version,
            'timestamp': datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            'training_samples': len(X_train),
            'n_features': X_train_vectorized.shape[1],
            'classes': model.classes_.tolist(),
            'model_path': shared_model_path,
            'vectorizer_path': shared_vectorizer_path
        }
        
        with open(f"{SHARED_MODELS_PATH}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Salvar no contexto
        context['task_instance'].xcom_push(key='model_version', value=model_version)
        context['task_instance'].xcom_push(key='model_path', value=latest_model_path)
        context['task_instance'].xcom_push(key='vectorizer_path', value=latest_vectorizer_path)
        
        logger.info(f"Modelo salvo: {latest_model_path}")
        logger.info(f"Vectorizer salvo: {latest_vectorizer_path}")
        
        # Avaliar no treino
        train_pred = model.predict(X_train_vectorized)
        train_accuracy = accuracy_score(y_train, train_pred)
        logger.info(f"Acurácia no treino: {train_accuracy:.4f}")
        
        return f"Model trained and saved: {model_version} (accuracy: {train_accuracy:.4f})"
    
    except Exception as e:
        logger.error(f"Erro no treinamento: {e!s}")
        raise

def evaluate_model(**context):
    """Avaliar performance do modelo"""
    try:
        # Carregar dados de teste
        X_test = pd.read_pickle('/tmp/X_test.pkl')
        y_test = pd.read_pickle('/tmp/y_test.pkl')
        
        # Carregar modelo
        model_path = context['task_instance'].xcom_pull(key='model_path', task_ids='train_model')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        
        # Vectorizar teste
        X_test_vectorized = vectorizer.transform(X_test)
        
        # Predições
        y_pred = model.predict(X_test_vectorized)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Salvar métricas no volume compartilhado
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'timestamp': datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            'n_test_samples': len(X_test)
        }
        
        with open(f"{SHARED_MODELS_PATH}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Acurácia no teste: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Salvar no contexto
        context['task_instance'].xcom_push(key='test_accuracy', value=str(accuracy))
        context['task_instance'].xcom_push(key='metrics_path', value=f"{SHARED_MODELS_PATH}/metrics.json")
        
        return f"Model evaluated with test accuracy: {accuracy:.4f}"
    
    except Exception as e:
        logger.error(f"Erro na avaliação: {e!s}")
        raise

# Definir DAG
dag = DAG(
    'medical_triage_training',
    default_args=default_args,
    description='Pipeline de treinamento do modelo de triagem médica',
    schedule='@weekly',
    catchup=False,
    tags=['medical', 'ml', 'training'],
    doc_md="""
    # Pipeline de Treinamento - Triagem Médica
    
    Esta DAG executa o pipeline completo de treinamento:
    1. Carrega dados do Parquet compartilhado
    2. Pré-processa os dados
    3. Treina o modelo Random Forest
    4. Avalia o modelo
    
    O modelo treinado é salvo no volume compartilhado `/shared/models/`
    para ser acessado pela API de inferência.
    """
)

# Tasks
start = EmptyOperator(
    task_id='start',
    dag=dag
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

end = EmptyOperator(
    task_id='end',
    dag=dag
)

# Definir fluxo
start >> load_data_task >> preprocess_task >> train_task >> evaluate_task >> end