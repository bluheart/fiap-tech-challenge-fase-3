
from pydantic import BaseModel, Field


class TextInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000, description="Medical text to classify")
    

class PredictionResponse(BaseModel):
    prediction: str
    raw_prediction: str
    confidence: float
    processing_time_ms: float
    model_version: str
    timestamp: str


class ModelStatus(BaseModel):
    model_loaded: bool
    model_version: str
    model_path: str | None = None
    last_loaded: str | None = None
    has_vectorizer: bool = False
    shared_path_exists: bool = False
    metadata: dict | None = None


class ReloadResponse(BaseModel):
    success: bool
    message: str
    model_version: str


class ONNXConversionResponse(BaseModel):
    success: bool
    message: str
    model_version: str
    onnx_path: str | None = None
    latest_onnx_path: str | None = None
    conversion_time_seconds: float | None = None
    input_name: str | None = None
    output_names: list[str] | None = None


class ONNXPredictionResponse(BaseModel):
    prediction: str
    raw_prediction: str
    confidence: float
    processing_time_ms: float
    model_version: str
    timestamp: str
    backend: str = "onnx"


class ONNXStatus(BaseModel):
    onnx_loaded: bool
    onnx_path: str | None = None
    model_version: str
    last_loaded: str | None = None
    output_names: list[str] = []
    metadata: dict | None = None


class BenchmarkResponse(BaseModel):
    sklearn_time_ms: float
    onnx_time_ms: float
    speedup: float
    sklearn_prediction: str
    onnx_prediction: str
    match: bool