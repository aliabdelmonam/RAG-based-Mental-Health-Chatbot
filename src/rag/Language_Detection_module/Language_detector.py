import sys
from pathlib import Path

# Ensure src directory is in path for module imports
# src_dir = str(Path(__file__).resolve().parents[2])  # 2 levels up to src/
# if src_dir not in sys.path:
    # sys.path.append(src_dir)

import joblib
import numpy as np
import re
from typing import Dict
from src.core.logger import get_logger  # Import the logger
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
    

    test_cases = [
    # --- ENGLISH ---
    {
        "text": "I have been feeling very anxious and stressed lately because of my exams.",
        "expected_lang": "en",
        "category": "Formal"
    },
    {
        "text": "idk what to do anymore im just so overwhelmed with everything rn",
        "expected_lang": "en",
        "category": "Informal / Slang"
    },
    {
        "text": "Panic attack",
        "expected_lang": "en",
        "category": "Short Input"
    },

    # --- ARABIC ---
    {
        "text": "أشعر بالقلق والتوتر الشديد في الآونة الأخيرة بسبب امتحاناتي.",
        "expected_lang": "ar",
        "category": "Formal (MSA)"
    },
    {
        "text": "مخنوق أوي ومش طايق نفسي اليومين دول وخايف من بكرة.",
        "expected_lang": "ar",
        "category": "Colloquial (Egyptian)"
    },
    {
        "text": "7ases eny mdaye2 awi w mesh 3aref anam",
        "expected_lang": "ar",
        "category": "Arabizi / Franco-Arabic"
    },

    # --- CHINESE ---
    {
        "text": "我最近因为考试感到非常焦虑和压力巨大。",
        "expected_lang": "zh",
        "category": "Formal (Simplified)"
    },
    {
        "text": "最近心好累，整个人emo了，感觉快崩溃了。",
        "expected_lang": "zh",
        "category": "Informal / Internet Slang (emo = emotional/sad)"
    },
    {
        "text": "焦虑症",
        "expected_lang": "zh",
        "category": "Short Input (Anxiety)"
    },

    # --- KOREAN ---
    {
        "text": "요즘 시험 때문에 너무 불안하고 스트레스를 많이 받고 있습니다.",
        "expected_lang": "ko",
        "category": "Formal (Honorific)"
    },
    {
        "text": "요새 기분 넘 우울하고 무기력해... 아무것도 하기 싫어 ㅠㅠ",
        "expected_lang": "ko",
        "category": "Informal / Text Slang (ㅠㅠ = crying eyes)"
    },
    {
        "text": "공황장애",
        "expected_lang": "ko",
        "category": "Short Input (Panic Disorder)"
    },

    # --- SPANISH ---
    {
        "text": "Me he sentido muy ansioso y estresado últimamente por mis exámenes.",
        "expected_lang": "es",
        "category": "Formal"
    },
    {
        "text": "estoy super bajoneado ultimamente y no tengo ganas de hacer nada",
        "expected_lang": "es",
        "category": "Informal / Dialect"
    },

    # --- FRENCH ---
    {
        "text": "Je me sens très anxieux et stressé ces derniers temps à cause de mes examens.",
        "expected_lang": "fr",
        "category": "Formal"
    },

    # --- GERMAN ---
    {
        "text": "Ich habe mich in letzter Zeit wegen meiner Prüfungen sehr ängstlich und gestresst gefühlt.",
        "expected_lang": "de",
        "category": "Formal"
    },

    # --- CODE-SWITCHING (EDGE CASES) ---
    {
        "text": "حاسس إني عندي panic attack مش قادر أتنفس",
        "expected_lang": "ar",
        "category": "Mixed Arabic/English"
    },
    {
        "text": "나 요즘 너무 stress 받아 진짜",
        "expected_lang": "ko",
        "category": "Mixed Korean/English"
    },

    {
        "text" :"what is the output of 2+2=?",
        "expected_lang": "en",
        "category": "Math"
    },
    {
        "text": "162+2+3+4++5+=4",
        "expected_lang": "en",
        "category": "Math"
    }
]
    passed_tests = 0
    for case in test_cases:
        prediction = lang_detector.predict(case["text"]) 
        status = "PASS" if prediction["language"] == case["expected_lang"] else "FAIL"
        if status == "PASS":
            passed_tests += 1
        print(f"Prediction: {prediction['language']}")
        print(f"Expected: {case['expected_lang']}")
        print(f"Status: {status}")
        print("-" * 20)
    print("Accuracy:" , {passed_tests/len(test_cases) * 100})