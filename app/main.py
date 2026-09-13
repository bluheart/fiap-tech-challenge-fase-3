# app/main.py
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import (
    BACKEND_COMPARISON,
    MODEL_LOADED,
    ONNX_MODEL_LOADED,
    ONNX_PREDICTION_TIME,
    ONNX_PREDICTIONS_TOTAL,
    setup_metrics,
)
from .model import MedicalTextClassifier
from .onnx_converter import ONNXMedicalTextClassifier, convert_sklearn_to_onnx
from .schemas import (
    BenchmarkResponse,
    ModelStatus,
    ONNXConversionResponse,
    ONNXPredictionResponse,
    ONNXStatus,
    PredictionResponse,
    ReloadResponse,
    TextInput,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Text Classifier API",
    description="API para classificação de textos médicos por urgência (sklearn + ONNX Runtime)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app)

MODEL_PATH = os.getenv("MODEL_PATH", None)
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", None)

# --- Carregar backend sklearn (original) ---
try:
    classifier = MedicalTextClassifier(MODEL_PATH) if MODEL_PATH else MedicalTextClassifier()
    logger.info("Modelo sklearn carregado com sucesso")
    MODEL_LOADED.set(1)
except Exception as e:
    logger.error(f"Falha ao carregar modelo sklearn: {e}")
    classifier = None
    MODEL_LOADED.set(0)

# --- Carregar backend ONNX (otimizado) ---
try:
    onnx_classifier = ONNXMedicalTextClassifier(ONNX_MODEL_PATH)
    if onnx_classifier.is_loaded():
        logger.info("Modelo ONNX carregado com sucesso")
        ONNX_MODEL_LOADED.set(1)
    else:
        logger.warning("Modelo ONNX não disponível. Use /convert-to-onnx para gerar.")
        ONNX_MODEL_LOADED.set(0)
except Exception as e:
    logger.error(f"Falha ao carregar modelo ONNX: {e}")
    onnx_classifier = None
    ONNX_MODEL_LOADED.set(0)


@app.get("/")
async def root():
    return {
        "message": "Medical Text Classifier API",
        "version": "1.1.0",
        "sklearn_status": "healthy" if classifier and classifier.model else "degraded",
        "onnx_status": "healthy" if onnx_classifier and onnx_classifier.is_loaded() else "degraded",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if classifier and classifier.model else "degraded",
        "model_loaded": bool(classifier and classifier.model),
        "onnx_loaded": bool(onnx_classifier and onnx_classifier.is_loaded()),
        "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
    }


@app.get("/model-status", response_model=ModelStatus)
async def get_model_status():
    if classifier is None:
        return ModelStatus(
            model_loaded=False,
            model_version="unknown",
            model_path=None,
            last_loaded=None,
            has_vectorizer=False,
        )
    return ModelStatus(**classifier.get_status())


@app.get("/onnx-status", response_model=ONNXStatus)
async def get_onnx_status():
    if onnx_classifier is None:
        return ONNXStatus(onnx_loaded=False, model_version="unknown")
    return ONNXStatus(**onnx_classifier.get_status())


@app.post("/reload-model", response_model=ReloadResponse)
async def reload_model():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    try:
        success = classifier.reload_model()
        MODEL_LOADED.set(1 if success else 0)
        return ReloadResponse(
            success=success,
            message="Model reloaded successfully" if success else "Failed to reload model",
            model_version=classifier.model_version if success else "unknown",
        )
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        MODEL_LOADED.set(0)
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Endpoint de predição sklearn (original) ----------
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    if classifier is None or classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = classifier.predict(input_data.text)
        BACKEND_COMPARISON.labels(backend="sklearn").observe(
            result["processing_time_ms"] / 1000.0
        )
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Endpoint de conversão para ONNX ----------
@app.post("/convert-to-onnx", response_model=ONNXConversionResponse)
async def convert_to_onnx():
    """
    Converte o modelo sklearn atual (RandomForest + TfidfVectorizer) para ONNX Runtime.
    Técnica de otimização de latência: exportação para ONNX, permitindo inferência
    mais rápida com grafo otimizado.
    """
    global onnx_classifier
    try:
        metadata = convert_sklearn_to_onnx()

        # Recarregar o classificador ONNX em memória
        onnx_classifier = ONNXMedicalTextClassifier(metadata["latest_onnx_path"])
        ONNX_MODEL_LOADED.set(1 if onnx_classifier.is_loaded() else 0)

        return ONNXConversionResponse(
            success=True,
            message="Modelo convertido para ONNX com sucesso",
            model_version=metadata["version"],
            onnx_path=metadata["onnx_path"],
            latest_onnx_path=metadata["latest_onnx_path"],
            conversion_time_seconds=metadata["conversion_time_seconds"],
            input_name=metadata["input_name"],
            output_names=metadata["output_names"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erro na conversão ONNX: {e}")
        ONNX_MODEL_LOADED.set(0)
        raise HTTPException(status_code=500, detail=f"Falha na conversão: {e}")


# ---------- Endpoint de predição ONNX (otimizado) ----------
@app.post("/predict-onnx", response_model=ONNXPredictionResponse)
async def predict_onnx(input_data: TextInput):
    """
    Inferência otimizada usando ONNX Runtime.
    Latência tipicamente menor que o backend sklearn.
    """
    if onnx_classifier is None or not onnx_classifier.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Modelo ONNX não carregado. Chame /convert-to-onnx primeiro.",
        )
    try:
        result = onnx_classifier.predict(input_data.text)

        ONNX_PREDICTION_TIME.observe(result["processing_time_ms"] / 1000.0)
        ONNX_PREDICTIONS_TOTAL.labels(prediction_class=result["prediction"]).inc()
        BACKEND_COMPARISON.labels(backend="onnx").observe(
            result["processing_time_ms"] / 1000.0
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro na predição ONNX: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Endpoint de benchmark sklearn vs ONNX ----------
@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(input_data: TextInput):
    """
    Compara a latência dos dois backends para o mesmo texto.
    Útil para demonstrar o ganho de performance da otimização ONNX.
    """
    if classifier is None or classifier.model is None:
        raise HTTPException(status_code=503, detail="Modelo sklearn não carregado")
    if onnx_classifier is None or not onnx_classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo ONNX não carregado")

    sklearn_result = classifier.predict(input_data.text)
    onnx_result = onnx_classifier.predict(input_data.text)

    sklearn_ms = sklearn_result["processing_time_ms"]
    onnx_ms = onnx_result["processing_time_ms"]
    speedup = round(sklearn_ms / onnx_ms, 2) if onnx_ms > 0 else 0.0

    return BenchmarkResponse(
        sklearn_time_ms=sklearn_ms,
        onnx_time_ms=onnx_ms,
        speedup=speedup,
        sklearn_prediction=sklearn_result["prediction"],
        onnx_prediction=onnx_result["prediction"],
        match=sklearn_result["prediction"] == onnx_result["prediction"],
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)