from src.rag.Language_Detection_module.Language_detector import LanguageDetector
from src.rag.Emotion_Classifier_module.Emotion_classifier import EmotionClassifier
from src.rag.Rag_module.rag_pipeline import RAGPipeline, RAGResult
from src.rag.Rag_module.full_pipeline import FullPipeline

# Defer IntentClassifier import to avoid circular dependency
def __getattr__(name):
    if name == 'IntentClassifier':
        from src.rag.Intent_Classifier_module.Intent_classifier import IntentClassifier
        return IntentClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
