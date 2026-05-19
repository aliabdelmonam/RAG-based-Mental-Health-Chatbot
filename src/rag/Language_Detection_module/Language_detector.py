import sys
from pathlib import Path

# Ensure src directory is in path for module imports
src_dir = str(Path(__file__).resolve().parents[2])  # 2 levels up to src/
if src_dir not in sys.path:
    sys.path.append(src_dir)

import joblib
import numpy as np
import re
from typing import Dict
from core.logger import get_logger  # Import the logger
import warnings

warnings.filterwarnings("ignore")

# Initialize logger for this module
logger = get_logger(f'LanguageDetector:')

class TextPreprocessor:
    @staticmethod
    def preprocess(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        # Remove punctuation, symbols, and operators (non-word and non-space characters)
        text = re.sub(r"[^\w\s]", "", text)
        # Remove digits and underscores to keep only letters
        text = re.sub(r"[\d_]", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

class LanguageDetector:
    def __init__(self, model_path: str, threshold: float = 0.60):
        logger.info("Initializing LanguageDetector...")
        self.model_path = model_path
        self.threshold = threshold
        self.model = self._load_model()
        logger.info("LanguageDetector initialized.")

    def _load_model(self):
        try:
            logger.debug("Loading model file...")
            model = joblib.load(self.model_path)
            logger.debug("Model file loaded.")
            return model
        except Exception as e:
            # Log the error with stack trace (exc_info=True)
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, text: str) -> Dict:
        logger.debug("Predicting language...")
        
        clean_text = TextPreprocessor.preprocess(text)

        if not clean_text:
            logger.warning("Empty text after preprocessing. Bypassing model prediction.")
            return {
                "language": None,
                "confidence": 0.0,
                "reliable": False
            }

        proba = self.model.predict_proba([clean_text])[0]
        classes = self.model.classes_

        top_idx = np.argmax(proba)
        top_conf = float(proba[top_idx])
        top_lang = classes[top_idx]

        if top_conf < self.threshold:
            logger.warning(f"Low confidence language prediction: {top_lang} ({top_conf:.2f})")
            return {
                "language": "uncertain",
                "confidence": top_conf,
                "reliable": False
            }

        logger.info(f"Language detected: {top_lang} ({top_conf:.2f})")
        return {
            "language": top_lang,
            "confidence": top_conf,
            "reliable": True
        }


if __name__ == "__main__":
    # This will automatically create a 'logs' folder and start writing to it
    lang_detector = LanguageDetector(model_path=r"C:\Users\BS\Downloads\language_detector.pkl", threshold=0.60)
    
    text = "عليا الطلاق الملك نمبر وان"
    result = lang_detector.predict(text)
    print(result)