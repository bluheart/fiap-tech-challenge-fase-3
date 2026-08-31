# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Medical Triage API", version="1.0.0")

# Métricas Prometheus
REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])
ERRORS = Counter("http_errors_total", "Total HTTP errors", ["method", "endpoint", "status"])


model = joblib.load("../models/model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")


class LaudoRequest(BaseModel):
    texto: str

class LaudoResponse(BaseModel):
    classificacao: str  # "normal", "atencao", "urgente"
    confianca: float


@app.post("/predict", response_model=LaudoResponse)
async def predict(request: LaudoRequest):
    REQUESTS.labels(method="POST", endpoint="/predict").inc()
    
    try:
        with LATENCY.labels(method="POST", endpoint="/predict").time():
            vectorized = vectorizer.transform([request.texto])
            proba = model.predict_proba(vectorized)[0]
            pred = model.classes_[np.argmax(proba)]
            
            return LaudoResponse(
                classificacao=pred,
                confianca=float(np.max(proba))
            )
    except Exception as e:
        ERRORS.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "healthy"}