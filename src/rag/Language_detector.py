import joblib
import numpy as np
import re
from typing import Dict


class TextPreprocessor:
    @staticmethod
    def preprocess(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        return text.lower()


class LanguageDetector:
    def __init__(self, model_path: str, threshold: float = 0.60):
        self.model_path = model_path
        self.threshold = threshold
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, text: str) -> Dict:
        clean_text = TextPreprocessor.preprocess(text)

        proba = self.model.predict_proba([clean_text])[0]
        classes = self.model.classes_

        top_idx = np.argmax(proba)
        top_conf = float(proba[top_idx])
        top_lang = classes[top_idx]

        if top_conf < self.threshold:
            return {
                "language": "uncertain",
                "confidence": top_conf,
                "reliable": False
            }

        return {
            "language": top_lang,
            "confidence": top_conf,
            "reliable": True
        }

lang_detector = LanguageDetector(model_path=r"C:\Users\BS\Downloads\language_detector.pkl", threshold=0.60)
text = "عليا الطلاق الملك نمبر وان"
result = lang_detector.predict(text)
print(result)