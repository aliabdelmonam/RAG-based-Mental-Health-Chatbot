import torch
import re
import nltk
import demoji
import sys
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Union
from pathlib import Path


# Ensure src directory is in path for module imports
src_dir = str(Path(__file__).resolve().parents[2])  # 2 levels up to src/
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.logger import get_logger

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


logger = get_logger(f'EmotionClassifier:')

# Emotion labels mapping
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

class TextPreprocessor:
    """
    Handles text cleaning: lowercasing, emoji conversion, and stopword removal.
    """
    _STOPWORDS = None 

    @classmethod
    def _get_stopwords(cls) -> set:
        """Loads NLTK stopwords on first use (lazy initialization)."""
        if cls._STOPWORDS is None:
            cls._STOPWORDS = set(nltk.corpus.stopwords.words('english'))
        return cls._STOPWORDS

    @staticmethod
    def preprocess(text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = demoji.replace_with_desc(text, sep=" ")
        stopwords = TextPreprocessor._get_stopwords()
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]

        return " ".join(filtered_words)


class EmotionClassifier:
    def __init__(self, model_path: str):
        """
        Initializes the Emotion Classifier by loading the model and tokenizer.
        
        Args:
            model_path (str): Path to the directory containing the fine-tuned model files.
        """
        logger.info(f"Initializing EmotionClassifier from: {model_path}")
        self.model_path = model_path
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path '{self.model_path}' does not exist.")
        except FileNotFoundError as e:
            logger.error(e)
            raise

        self.device = self._get_device()
        
        self.tokenizer, self.model = self._load_model()
        # Use custom emotion labels instead of model's id2label
        self.id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
        
        logger.info(f"Model loaded successfully. Labels: {list(self.id2label.values())}")
        logger.info(f"Running on device: {self.device}")

    def _get_device(self) -> str:
        """Determines the device (CUDA GPU or CPU) to run the model on."""
        if torch.cuda.is_available():
            logger.debug("CUDA detected. Using GPU.")
            return "cuda"
        logger.debug("CUDA not detected. Using CPU.")
        return "cpu"

    def _load_model(self):
        """Loads the tokenizer and model from the local directory."""
        try:
            logger.debug("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            logger.debug("Loading model weights...")
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Move model to the appropriate device (GPU/CPU)
            model.to(self.device)
            
            # Set model to evaluation mode (disables dropout, etc.)
            model.eval()
            
            return tokenizer, model
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def predict(self, text: str) -> Dict:
        """
        Predicts the emotion of the input text.
        
        Args:
            text (str): The input text to classify.
            
        Returns:
            Dict: Contains 'emotion', 'confidence', and 'all_scores'.
        """
        original_text_len = len(text) if text else 0
        logger.debug(f"Predicting emotion for: '{text[:30]}...'")
        
        if not isinstance(text, str) or text.strip() == "":
            logger.warning("Empty or invalid text received.")
            return {"emotion": None, "confidence": 0.0, "all_scores": {}}

        try:
            # Apply Preprocessing
            clean_text = TextPreprocessor.preprocess(text)
            
            # If preprocessing removes everything (e.g. only stopwords), handle gracefully
            if not clean_text.strip():
                logger.warning("Text became empty after preprocessing. Using original text.")
                clean_text = text.lower()

            # 1. Tokenize input
            inputs = self.tokenizer(
                clean_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 2. Inference (disable gradient calculation for speed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # 3. Process outputs (Softmax to get probabilities)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Move results back to CPU for numpy/python handling
            probs_list = probabilities.cpu().numpy()[0]
            
            # Get top prediction
            top_idx = int(torch.argmax(probabilities, dim=-1))
            top_conf = float(probs_list[top_idx])
            top_emotion = self.id2label[top_idx]

            # Format all scores for detailed view
            all_scores = {self.id2label[i]: float(p) for i, p in enumerate(probs_list)}

            logger.info(f"Prediction: {top_emotion} ({top_conf:.4f})")

            return {
                "emotion": top_emotion,
                "confidence": top_conf,
                "all_scores": all_scores
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return {"emotion": "error", "confidence": 0.0, "all_scores": {}}


if __name__ == "__main__":
    # Example Usage
    # Point this to the folder where you unzipped your model
    MODEL_DIR = r"D:\ITI\roberta_multilabel_model" 
    
    try:
        classifier = EmotionClassifier(model_path=MODEL_DIR)
        
        sample_text = "my boss fired me today 😭" # Emoji test
        result = classifier.predict(sample_text)
        print(f"Input: {sample_text}")
        print(f"Result: {result}")
        
        sample_text_2 = "I am not happy with the service" # Stopword test
        result_2 = classifier.predict(sample_text_2)
        print(f"Input: {sample_text_2}")
        print(f"Result: {result_2}")
        
    except Exception as e:
        print(f"Critical error in main execution: {e}")