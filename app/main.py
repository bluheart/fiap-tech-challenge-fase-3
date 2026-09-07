from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import logging
from .model import MedicalTextClassifier
from .schemas import TextInput, PredictionResponse
from .metrics import setup_metrics
import joblib
import os

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

# Model path
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/classifier.joblib")

# Load model
try:
    classifier = MedicalTextClassifier(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    classifier = None

@app.get("/")
async def root():
    return {"message": "Medical Text Classifier API", "status": "healthy"}

@app.get("/health")
async def health_check():
    if classifier is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = classifier.predict(input_data.text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)