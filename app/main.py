import pickle
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

app = FastAPI(title="Medical Triage API")

# Métricas Prometheus
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['classification'])

class MedicalReport(BaseModel):
    text: str
    patient_id: str | None = None

class PredictionResponse(BaseModel):
    classification: str
    confidence: float
    latency_ms: float
    model_version: str

# Carregar modelo
try:
    with open('models/original_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        MODEL = model_data['model']
        VECTORIZER = model_data['vectorizer']
        MODEL_VERSION = model_data.get('version', '1.0.0')
except FileNotFoundError:
    MODEL = None
    VECTORIZER = None
    MODEL_VERSION = 'not_loaded'

@app.get("/")
def read_root():
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict", response_model=PredictionResponse)
async def predict(report: MedicalReport):
    start_time = time.time()
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Pré-processamento
        text_vectorized = VECTORIZER.transform([report.text]) # type: ignore
        
        # Predição
        prediction = MODEL.predict(text_vectorized)[0]
        probability = np.max(MODEL.predict_proba(text_vectorized)[0])
        
        latency = (time.time() - start_time) * 1000
        
        # Registrar métricas
        PREDICTION_COUNT.labels(classification=prediction).inc()
        REQUEST_DURATION.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        
        return PredictionResponse(
            classification=prediction,
            confidence=float(probability),
            latency_ms=round(latency, 2),
            model_version=MODEL_VERSION
        )
    except Exception as e: #noqa: BLE001
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))