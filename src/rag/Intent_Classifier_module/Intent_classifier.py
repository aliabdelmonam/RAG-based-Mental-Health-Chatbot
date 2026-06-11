from __future__ import annotations

import re
import sys
import json
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List, TYPE_CHECKING

from src.core.Config import get_settings
from src.core.logger import get_logger
from src.rag.Language_Detection_module.Language_detector import LanguageDetector
from src.stores import GenerationConfig
from src.prompts import intent_system_prompt
from src.stores import LLMProviderFactory,GenerationConfig
from langchain_core.messages import BaseMessage

if TYPE_CHECKING:
    from src.rag.Rag_module.base_pipeline import BundleManager

settings = get_settings()
logger = get_logger(f"IntentClassifier:")

_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "Intent_classifier_prompt.txt"
_FALLBACK_PROMPT = (
    "You are an intent classifier for a mental health chatbot.\n"
    "Classify the user message into exactly one of:\n"
    "greeting | goodbye | gratitude | asking_mental_health_question | out_of_scope | crisis\n"
    'Return ONLY valid JSON: {"intent": "<label>"}'
)


class IntentLabel(str, Enum):
    GREETING                      = "greeting"
    GOODBYE                       = "goodbye"
    GRATITUDE                     = "gratitude"
    ASKING_MENTAL_HEALTH_QUESTION = "asking_mental_health_question"
    OUT_OF_SCOPE                  = "out_of_scope"
    CRISIS                        = "crisis"


@dataclass
class IntentResult:
    intent:       IntentLabel
    raw_response: str
    requires_rag: bool


class IntentClassifier:
    """LLM-based intent classifier for a multilingual mental health chatbot."""

    _RAG_INTENTS       = {IntentLabel.ASKING_MENTAL_HEALTH_QUESTION, IntentLabel.CRISIS}  # intents that should trigger RAG retrieval
    _GENERATION_CONFIG = GenerationConfig(temperature=0.0, max_new_tokens=6000)

    def __init__(
        self,
        client: BundleManager,
        language_detector: Optional[LanguageDetector] = None,
    ) -> None:
        self.client = client
        self._language_detector = language_detector

        try:
            self._system_prompt = intent_system_prompt
        except FileNotFoundError:
            logger.warning("Prompt file not found at %s — using fallback.", _PROMPT_PATH)
            self._system_prompt = _FALLBACK_PROMPT

        logger.info("IntentClassifier ready.")


    def classify(self, user_message: str,chat_history: List[BaseMessage], detected_language: Optional[str] = None) -> IntentResult:
        """Classify the intent of a user message, with optional language hint."""
        if not user_message or not user_message.strip():
            return self._fallback("empty input")
        
        if detected_language is None and self._language_detector is not None:
            lang_result = self._language_detector.predict(user_message)
            if lang_result.get("reliable", False):
                detected_language = lang_result["language"]

        messages = self._system_prompt.format_messages(
            detected_language= detected_language or "unknown",
            user_message= user_message,
            recent_context=self._format_recent_context(chat_history=chat_history, k=2)
        )

        response = self.client.generate(
            messages=messages,
            config=self._GENERATION_CONFIG,
        )

        raw = response.content
        logger.debug("LLM raw response: %s", raw)
        return self._parse(raw)

    def health_check(self) -> bool:
        return self.client.health_check()

    def _parse(self, raw: Any) -> IntentResult:
        try:
            # Normalize raw LLM output and strip markdown fences, then grab the first {...} block
            raw_text = self._normalize_raw(raw)
            cleaned = re.sub(r"```(?:json)?\s*|```", "", raw_text).strip()
            match   = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
            data    = json.loads(match.group() if match else cleaned)
    
            intent_str = data.get("intent", "").strip().lower()
            try:
                intent = IntentLabel(intent_str)
            except ValueError:
                logger.warning("Unknown intent label '%s' — defaulting to out_of_scope.", intent_str)
                intent = IntentLabel.OUT_OF_SCOPE
    
            requires_rag = intent in self._RAG_INTENTS
    
            logger.info("Classified intent: %s", intent.value)
            return IntentResult(intent=intent, raw_response=raw, requires_rag=requires_rag)

        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            logger.error("Failed to parse LLM response '%s': %s", raw, exc)
            return self._fallback(raw)

    @staticmethod
    def _normalize_raw(raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return json.dumps(raw)
        if isinstance(raw, list):
            if all(isinstance(item, str) for item in raw):
                return "\n".join(raw)
            return json.dumps(raw)
        return str(raw)

    @staticmethod
    def _fallback(raw: Any) -> IntentResult:
        return IntentResult(intent=IntentLabel.OUT_OF_SCOPE, raw_response=str(raw), requires_rag=False)

    def _format_recent_context(self,chat_history: List[BaseMessage],k: int=2) -> str:
        if not chat_history:
            return "No recent messages."
        turns = chat_history[-(k * 2):]  # k user + k assistant turns

        lines = []
        for m in turns:
            role = "User" if m.type == "human" else "Assistant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

if __name__ == "__main__":
    LANG_MODEL_PATH = r"C:\Users\aliab\Downloads\language_detector.pkl"

    try:
        lang_detector = LanguageDetector(model_path=LANG_MODEL_PATH, threshold=0.60)
    except Exception as e:
        logger.warning("Could not load LanguageDetector (%s) — continuing without it.", e)
        lang_detector = None
    llm_provider = LLMProviderFactory(settings)
    generation_client = llm_provider.create(provider=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

    classifier = IntentClassifier(generation_client=generation_client, language_detector=lang_detector)
    test_cases = [
        "hello, how are you?",
        "I feel anxious all the time",
        "شكرا جدا على مساعدتك",
        "what's the weather like tomorrow?",
        "bye, take care!",
        "انا حاسس بقلق وما قادر انام",
        "I want to hurt myself",
        "2 + 2 = ?",
    ]

    print(f"\n{'USER MESSAGE':<40} {'INTENT':<35} {'RAG'}")
    print("-" * 80)
    for text in test_cases:
        r = classifier.classify(text, chat_history=[])
        print(f"{text:<40} {r.intent.value:<35} {r.requires_rag}")
        print(f"LLM Response: {r}")
    print("--- Testing IntentClassifier Setup ---")
    
        
    print("To fully test classification here, pass your generation_client client to the class.")