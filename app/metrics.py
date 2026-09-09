import time

from fastapi import FastAPI, Request
from prometheus_client import Counter, Gauge, Histogram

# Define metrics
REQUESTS_TOTAL = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ERRORS_TOTAL = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['endpoint', 'status_code']
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

# Novas métricas específicas do modelo
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['prediction_class']
)

PREDICTION_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Prediction confidence distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

MODEL_PREDICTION_TIME = Histogram(
    'model_prediction_time_seconds',
    'Time taken for model prediction',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)


def setup_metrics(app: FastAPI):
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        endpoint = request.url.path
        
        # Increment request count
        REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=request.method
        ).inc()
        
        # Process request
        response = await call_next(request)
        
        # Record duration
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        
        # Record errors
        if response.status_code >= 400:
            ERRORS_TOTAL.labels(
                endpoint=endpoint,
                status_code=str(response.status_code)
            ).inc()
        
        return response