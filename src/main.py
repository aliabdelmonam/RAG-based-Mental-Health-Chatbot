from rag import IntentClassifier, LanguageDetector, EmotionClassifier
from stores import LLMProviderFactory
from core import get_settings

settings = get_settings()
# Initialize LLM provider factory with config
llm_provider = LLMProviderFactory(settings)

# create LLM provider
generation_client = llm_provider.create(provider=settings.GENERATION_BACKEND)

# Initialize intent classifier with the generation client and language detector
LanguageDetector =  LanguageDetector(model_path=r"C:\Users\BS\Downloads\language_detector.pkl", threshold=0.60)
intent_cls = IntentClassifier(generation_client=generation_client, language_detector=LanguageDetector)


# intent_cls.health_check()
generation_client.health_check()
generation_client.set_generation_model(settings.GENERATION_MODEL_ID)
result = intent_cls.classify('I have been feeling very anxious and stressed lately because of my exams.')

# print(result)