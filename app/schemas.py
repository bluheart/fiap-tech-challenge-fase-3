
from pydantic import BaseModel, Field


class TextInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000, description="Medical text to classify")
    
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time_ms: float
    model_version: str
    timestamp: str