import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pytz

logger = logging.getLogger(__name__)


class MedicalTextClassifier:
    def __init__(self, model_path = None):
        """
        Inicializa o classificador com suporte a múltiplos formatos de modelo
        
        Args:
            model_path: Caminho para o modelo. Se None, tenta carregar do volume compartilhado.
        """
        self.model = None
        self.vectorizer = None
        self.model_version = "unknown"
        self.model_path = None
        self.last_loaded = None
        
        # Caminhos possíveis para o modelo
        self.shared_models_path = '/shared/models'
        self.shared_vectorizers_path = '/shared/vectorizers'
        self.local_models_path = '/app/models'
        
        # Tentar carregar o modelo
        if model_path and os.path.exists(model_path):
            self._load_model_from_path(model_path)
        else:
            self._load_best_available_model()
    
    def _load_best_available_model(self):
        """Carrega o melhor modelo disponível (compartilhado ou local)"""
        # Prioridade 1: Modelo compartilhado mais recente
        shared_model = f"{self.shared_models_path}/latest_model.pkl"
        if os.path.exists(shared_model):
            self._load_model_from_path(shared_model)
            return
        
        # Prioridade 2: Modelo compartilhado versionado (mais recente)
        if os.path.exists(self.shared_models_path):
            model_files = list(Path(self.shared_models_path).glob("*_model.pkl"))
            if model_files:
                # Pegar o mais recente
                latest_model = max(model_files, key=os.path.getmtime)
                self._load_model_from_path(str(latest_model))
                return
        
        # Prioridade 3: Modelo local
        local_models = [
            f"{self.local_models_path}/classifier.joblib",
            f"{self.local_models_path}/original_model.pkl",
            f"{self.local_models_path}/latest_model.pkl"
        ]
        
        for model_path in local_models:
            if os.path.exists(model_path):
                self._load_model_from_path(model_path)
                return
        
        logger.warning("Nenhum modelo encontrado. API funcionará sem modelo.")
    
    def _load_model_from_path(self, model_path: str):
        """Carrega modelo de diferentes formatos"""
        try:
            logger.info(f"Carregando modelo de: {model_path}")
            
            # Verificar extensão do arquivo
            file_ext = Path(model_path).suffix.lower()
            
            if file_ext == '.joblib':
                # Modelo no formato joblib
                self.model = joblib.load(model_path)
                self.model_version = "1.0.0"
                
            elif file_ext == '.pkl':
                # Modelo no formato pickle (possivelmente com vectorizer)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Verificar se é um dicionário com model e vectorizer
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.vectorizer = model_data.get('vectorizer')
                    self.model_version = model_data.get('version', '1.0.0')
                else:
                    # Modelo simples
                    self.model = model_data
                    self.model_version = "1.0.0"
            
            else:
                raise ValueError(f"Formato de modelo não suportado: {file_ext}")
            
            self.model_path = model_path
            self.last_loaded = datetime.now(pytz.timezone('America/Sao_Paulo'))
            
            # Carregar metadados se disponíveis
            self._load_metadata()
            
            logger.info(f"Modelo carregado com sucesso. Versão: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _load_metadata(self):
        """Carrega metadados do modelo se disponíveis"""
        metadata_paths = [
            f"{self.shared_models_path}/metadata.json",
            f"{self.local_models_path}/metadata.json"
        ]
        
        for metadata_path in metadata_paths:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Metadados carregados: {self.metadata.get('version', 'unknown')}")
                    return
                except Exception as e:
                    logger.warning(f"Erro ao carregar metadados: {e}")
        
        self.metadata = None
    
    def reload_model(self):
        """Recarrega o modelo do disco"""
        logger.info("Recarregando modelo...")
        try:
            self._load_best_available_model()
            return True
        except Exception as e:
            logger.error(f"Erro ao recarregar modelo: {e}")
            return False
    
    def predict(self, text: str) -> dict:
        """
        Faz predição com o modelo carregado
        
        Args:
            text: Texto médico para classificar
            
        Returns:
            Dicionário com predição, confiança e metadados
        """
        if self.model is None:
            raise ValueError("Modelo não carregado")
        
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess(text)
        
        # Vectorizar se necessário
        if self.vectorizer is not None:
            text_vectorized = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_vectorized)[0]
            
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(text_vectorized)[0]
                confidence = float(np.max(probability))
            else:
                confidence = 1.0
        else:
            # Modelo que aceita texto diretamente
            prediction = self.model.predict([processed_text])[0]
            
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba([processed_text])[0]
                confidence = float(np.max(probability))
            else:
                confidence = 1.0
        
        # Mapear classe para label legível
        prediction_label = self._map_prediction_label(prediction)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        brazil_tz = pytz.timezone('America/Sao_Paulo')
        
        return {
            "prediction": prediction_label,
            "raw_prediction": str(prediction),
            "confidence": confidence,
            "processing_time_ms": round(processing_time, 2),
            "model_version": self.model_version,
            "timestamp": datetime.now(brazil_tz).isoformat()
        }
    
    def _map_prediction_label(self, prediction) -> str:
        """
        Mapeia a predição numérica para label legível
        
        Args:
            prediction: Valor da predição
            
        Returns:
            Label legível da classe
        """
        # Mapeamento comum para triagem médica
        class_mapping = {
            0: 'normal',
            1: 'urgente'
        }
        
        # Tentar mapear diretamente
        if isinstance(prediction, str):
            prediction_lower = prediction.lower()
            if prediction_lower in class_mapping:
                return class_mapping[prediction_lower]
            return prediction_lower
        
        # Para valores numéricos
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            prediction_key = int(prediction)
            if prediction_key in class_mapping:
                return class_mapping[prediction_key]
        
        return str(prediction)
    
    def _preprocess(self, text: str) -> str:
        """
        Pré-processa o texto antes da classificação
        
        Args:
            text: Texto original
            
        Returns:
            Texto pré-processado
        """
        # Conversão para lowercase
        text = text.lower()
        
        # Remover caracteres especiais (se necessário)
        # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remover espaços extras
        text = ' '.join(text.split())
        
        return text.strip()
    
    def get_status(self) -> dict:
        """
        Retorna o status atual do modelo
        
        Returns:
            Dicionário com informações do modelo
        """
        return {
            'model_loaded': self.model is not None,
            'model_version': self.model_version,
            'model_path': self.model_path,
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
            'has_vectorizer': self.vectorizer is not None,
            'shared_path_exists': os.path.exists(self.shared_models_path),
            'metadata': self.metadata
        }