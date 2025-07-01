import numpy as np
from typing import Union, Tuple

class LiquidVolumePredictor:
    """Inference class for liquid volume prediction"""
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        # Load model weights
        pass
    
    def predict_volume(self, video_path: str) -> float:
        """Predict liquid volume from video"""
        # Process video and return volume estimate
        pass
    
    def predict_from_frame(self, frame: np.ndarray) -> float:
        """Predict liquid volume from single frame"""
        # Process frame and return volume estimate
        pass