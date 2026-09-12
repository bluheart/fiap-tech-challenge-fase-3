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
    def __init__(self, model_path=None):
        self.model = None
        self.vectorizer = None
        self.model_version = "unknown"
        self.model_path = None
        self.last_loaded = None
        self.metadata = None

        self.shared_models_path = '/shared/models'
        self.shared_vectorizers_path = '/shared/vectorizers'
        self.local_models_path = '/app/models'

        if model_path and os.path.exists(model_path):
            self._load_model_from_path(model_path)
        else:
            self._load_best_available_model()

    def _load_best_available_model(self):
        """Carrega o melhor modelo disponível (compartilhado ou local)"""
        # Prioridade 1: latest_model.pkl do volume compartilhado
        shared_latest = f"{self.shared_models_path}/latest_model.pkl"
        if os.path.exists(shared_latest):
            self._load_model_from_path(shared_latest)
            return
        logger.warning("Nenhum modelo encontrado. API funcionará sem modelo.")

    def _load_model_from_path(self, model_path: str):
        """Carrega modelo dos formatos gerados pela DAG."""
        try:
            logger.info(f"Carregando modelo de: {model_path}")
            file_ext = Path(model_path).suffix.lower()

            if file_ext == '.joblib':
                self.model = joblib.load(model_path)
                self.model_version = "1.0.0"

            elif file_ext == '.pkl':
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                if isinstance(model_data, dict):
                    # Formato salvo pela DAG: {'model', 'vectorizer', 'version', ...}
                    self.model = model_data.get('model')
                    self.vectorizer = model_data.get('vectorizer')
                    self.model_version = model_data.get('version', '1.0.0')

                    # Fallback: se vectorizer não veio no dict, buscar arquivo separado
                    if self.vectorizer is None:
                        self._try_load_shared_vectorizer()
                else:
                    # Modelo "puro" (ex.: sklearn estimator)
                    self.model = model_data
                    self.model_version = "1.0.0"
                    self._try_load_shared_vectorizer()
            else:
                raise ValueError(f"Formato de modelo não suportado: {file_ext}")

            self.model_path = model_path
            self.last_loaded = datetime.now(pytz.timezone('America/Sao_Paulo'))

            self._load_metadata()
            logger.info(
                f"Modelo carregado. Versão: {self.model_version} | "
                f"vectorizer: {'sim' if self.vectorizer else 'não'}"
            )

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def _try_load_shared_vectorizer(self):
        """Tenta carregar o vectorizer de /shared/vectorizers/latest_vectorizer.pkl"""
        candidates = [
            f"{self.shared_vectorizers_path}/latest_vectorizer.pkl",
        ]
        # Também tenta o vectorizer versionado correspondente
        if self.model_path:
            version = Path(self.model_path).stem.replace("_model", "")
            candidates.append(f"{self.shared_vectorizers_path}/{version}_vectorizer.pkl")

        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    logger.info(f"Vectorizer carregado de: {path}")
                    return
                except Exception as e:
                    logger.warning(f"Falha ao carregar vectorizer {path}: {e}")

        logger.warning("Nenhum vectorizer encontrado no volume compartilhado.")

    def _load_metadata(self):
        for metadata_path in [
            f"{self.shared_models_path}/metadata.json",
            f"{self.local_models_path}/metadata.json",
        ]:
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
        logger.info("Recarregando modelo...")
        try:
            self.model = None
            self.vectorizer = None
            self._load_best_available_model()
            return self.model is not None
        except Exception as e:
            logger.error(f"Erro ao recarregar modelo: {e}")
            return False

    def predict(self, text: str) -> dict:
        if self.model is None:
            raise ValueError("Modelo não carregado")

        start_time = time.time()
        processed_text = self._preprocess(text)

        if self.vectorizer is not None:
            text_vectorized = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_vectorized)[0]
            confidence = (
                float(np.max(self.model.predict_proba(text_vectorized)[0]))
                if hasattr(self.model, 'predict_proba') else 1.0
            )
        else:
            prediction = self.model.predict([processed_text])[0]
            confidence = (
                float(np.max(self.model.predict_proba([processed_text])[0]))
                if hasattr(self.model, 'predict_proba') else 1.0
            )

        prediction_label = self._map_prediction_label(prediction)
        processing_time = (time.time() - start_time) * 1000
        brazil_tz = pytz.timezone('America/Sao_Paulo')

        return {
            "prediction": prediction_label,
            "raw_prediction": str(prediction),
            "confidence": confidence,
            "processing_time_ms": round(processing_time, 2),
            "model_version": self.model_version,
            "timestamp": datetime.now(brazil_tz).isoformat(),
        }

    def _map_prediction_label(self, prediction) -> str:
        class_mapping = {0: 'normal', 1: 'urgente'}

        if isinstance(prediction, str):
            key = prediction.lower()
            return class_mapping.get(key, key) # type: ignore

        if isinstance(prediction, (int, float, np.integer, np.floating)):
            return class_mapping.get(int(prediction), str(prediction))

        return str(prediction)

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = ' '.join(text.split())
        return text.strip()

    def get_status(self) -> dict:
        return {
            'model_loaded': self.model is not None,
            'model_version': self.model_version,
            'model_path': self.model_path,
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
            'has_vectorizer': self.vectorizer is not None,
            'shared_path_exists': os.path.exists(self.shared_models_path),
            'metadata': self.metadata,
        }