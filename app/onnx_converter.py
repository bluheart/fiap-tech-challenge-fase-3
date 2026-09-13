# app/onnx_converter.py
"""
Conversão do modelo sklearn (RandomForest + TfidfVectorizer) para ONNX Runtime.
A técnica aplicada é a exportação para ONNX Runtime, que permite inferência
mais rápida em CPU, com grafo otimizado e sem overhead do Python/sklearn.
"""
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)

SHARED_MODELS_PATH = "/shared/models"
SHARED_VECTORIZERS_PATH = "/shared/vectorizers"
ONNX_MODELS_PATH = "/shared/onnx"


def _ensure_dirs():
    for p in [ONNX_MODELS_PATH, SHARED_MODELS_PATH, SHARED_VECTORIZERS_PATH]:
        Path(p).mkdir(parents=True, exist_ok=True)


def convert_sklearn_to_onnx(
    model_path: str | None = None,
    vectorizer_path: str | None = None,
    output_dir: str = ONNX_MODELS_PATH,
) -> dict:
    """
    Converte o modelo sklearn (RandomForest) + TfidfVectorizer para ONNX.

    Estratégia:
      - O TfidfVectorizer é convertido para um grafo ONNX via skl2onnx.
      - O RandomForestClassifier é convertido para ONNX via skl2onnx.
      - Encadeamos os dois em um único modelo ONNX (pipeline) usando
        skl2onnx.common.data_types + to_onnx do sklearn-onnx, ou
        concatenamos em um Pipeline e exportamos.

    Retorna dict com metadados da conversão.
    """
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType
    from sklearn.pipeline import Pipeline

    _ensure_dirs()

    # 1) Descobrir caminhos
    if model_path is None:
        model_path = f"{SHARED_MODELS_PATH}/latest_model.pkl"
    if vectorizer_path is None:
        vectorizer_path = f"{SHARED_VECTORIZERS_PATH}/latest_vectorizer.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # 2) Carregar modelo e vectorizer
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    if isinstance(model_data, dict):
        model = model_data.get("model")
        vectorizer = model_data.get("vectorizer")
        version = model_data.get("version", "unknown")
    else:
        model = model_data
        vectorizer = None
        version = "unknown"

    # Fallback: vectorizer separado
    if vectorizer is None and os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

    if model is None:
        raise ValueError("Modelo sklearn inválido (None).")
    if vectorizer is None:
        raise ValueError("Vectorizer não encontrado. Necessário para exportar pipeline.")

    # 3) Montar Pipeline sklearn (vectorizer -> modelo)
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", model),
    ])

    # 4) Definir tipo de entrada: texto (string)
    initial_types = [("text", StringTensorType([None, 1]))]

    # 5) Converter para ONNX
    logger.info("Convertendo pipeline sklearn -> ONNX ...")
    t0 = time.time()
    onnx_model = to_onnx(
        pipeline,
        initial_types=initial_types,
        options={id(model): {"zipmap": False}},  # saída como tensor, sem ZipMap
    )
    conversion_time = time.time() - t0

    # 6) Salvar arquivo ONNX
    onnx_filename = f"{version}_model.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Também salvar como "latest" para a API
    latest_onnx_path = os.path.join(output_dir, "latest_model.onnx")
    with open(latest_onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 7) Metadados da conversão
    metadata = {
        "version": version,
        "onnx_path": onnx_path,
        "latest_onnx_path": latest_onnx_path,
        "conversion_time_seconds": round(conversion_time, 4),
        "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "input_name": "text",
        "output_names": [o.name for o in onnx_model.graph.output],
        "opset": onnx_model.opset_import[0].version if onnx_model.opset_import else None,
        "source_model": model_path,
        "source_vectorizer": vectorizer_path,
    }

    metadata_path = os.path.join(output_dir, "onnx_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Modelo ONNX salvo em: {onnx_path} ({conversion_time:.3f}s)")
    return metadata


class ONNXMedicalTextClassifier:
    """
    Classificador otimizado usando ONNX Runtime.
    A inferência é feita diretamente no grafo ONNX (vectorizer + RandomForest),
    eliminando o overhead do Python/sklearn e acelerando a predição.
    """

    def __init__(self, onnx_path: str | None = None):

        self.onnx_path = onnx_path or f"{ONNX_MODELS_PATH}/latest_model.onnx"
        self.session = None
        self.model_version = "unknown"
        self.last_loaded = None
        self.metadata = None

        if os.path.exists(self.onnx_path):
            self._load()
        else:
            logger.warning(f"Modelo ONNX não encontrado: {self.onnx_path}")

    def _load(self):
        import onnxruntime as ort

        logger.info(f"Carregando modelo ONNX: {self.onnx_path}")
        # Otimizações de sessão para latência
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # evita contenção em inferência pequena
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Metadados
        meta_path = os.path.join(os.path.dirname(self.onnx_path), "onnx_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.model_version = self.metadata.get("version", "unknown")
        else:
            self.model_version = Path(self.onnx_path).stem

        self.last_loaded = datetime.now(ZoneInfo("America/Sao_Paulo"))
        logger.info(f"ONNX carregado. Versão: {self.model_version} | outputs={self.output_names}")

    def is_loaded(self) -> bool:
        return self.session is not None

    def predict(self, text: str) -> dict:
        if self.session is None:
            raise ValueError("Modelo ONNX não carregado")

        start = time.time()

        # Pré-processamento mínimo (o vectorizer do ONNX já faz o TF-IDF)
        processed = " ".join(text.lower().split()).strip()

        # ONNX espera shape [N, 1] de strings
        input_array = np.array([[processed]], dtype=object)

        outputs = self.session.run(self.output_names, {self.input_name: input_array})

        # output_names típicos: ['label', 'probabilities']
        label = outputs[0][0]
        probs = outputs[1][0] if len(outputs) > 1 else None

        confidence = float(np.max(probs)) if probs is not None else 1.0
        prediction_label = self._map_prediction_label(label)

        processing_time = (time.time() - start) * 1000
        brazil_tz = ZoneInfo("America/Sao_Paulo")

        return {
            "prediction": prediction_label,
            "raw_prediction": str(label),
            "confidence": confidence,
            "processing_time_ms": round(processing_time, 2),
            "model_version": self.model_version,
            "timestamp": datetime.now(brazil_tz).isoformat(),
            "backend": "onnx",
        }

    @staticmethod
    def _map_prediction_label(prediction) -> str:
        class_mapping = {0: "normal", 1: "urgente"}
        if isinstance(prediction, str):
            key = prediction.lower()
            return class_mapping.get(key, key)  # type: ignore
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            return class_mapping.get(int(prediction), str(prediction))
        return str(prediction)

    def get_status(self) -> dict:
        return {
            "onnx_loaded": self.session is not None,
            "onnx_path": self.onnx_path,
            "model_version": self.model_version,
            "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
            "output_names": getattr(self, "output_names", []),
            "metadata": self.metadata,
        }