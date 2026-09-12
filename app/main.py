import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import MODEL_LOADED, setup_metrics
from .model import MedicalTextClassifier
from .schemas import ModelStatus, PredictionResponse, ReloadResponse, TextInput

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Text Classifier API",
    description="API for classifying medical texts by urgency",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup metrics
setup_metrics(app)

# Model path (opcional, se não definido, usa caminhos compartilhados)
MODEL_PATH = os.getenv("MODEL_PATH", None)

# Load model
try:
    classifier = MedicalTextClassifier(MODEL_PATH) if MODEL_PATH else MedicalTextClassifier()
    logger.info("Model loaded successfully")
    MODEL_LOADED.set(1)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    classifier = None
    MODEL_LOADED.set(0)


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Medical Text Classifier API",
        "status": "healthy" if classifier and classifier.model else "degraded",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if classifier is None or classifier.model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": classifier.model_version,
        "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()
    }


@app.get("/model-status", response_model=ModelStatus)
async def get_model_status():
    """Get detailed model status"""
    if classifier is None:
        return ModelStatus(
            model_loaded=False,
            model_version="unknown",
            model_path=None,
            last_loaded=None,
            has_vectorizer=False
        )
    
    status = classifier.get_status()
    return ModelStatus(**status)


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
            model_version=classifier.model_version if success else "unknown"
        )
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        MODEL_LOADED.set(0)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """Make prediction endpoint"""
    if classifier is None or classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = classifier.predict(input_data.text)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)