import joblib
import time
import numpy as np
from datetime import datetime

class MedicalTextClassifier:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_version = "1.0.0"
        
    def predict(self, text: str) -> dict:
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess(text)
        
        # Make prediction
        prediction = self.model.predict([processed_text])[0]
        probability = self.model.predict_proba([processed_text])[0]
        confidence = float(np.max(probability))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "prediction": str(prediction),
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat()
        }
    
    def _preprocess(self, text: str) -> str:
        # Add your preprocessing logic here
        return text.lower().strip()