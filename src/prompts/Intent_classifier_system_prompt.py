from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

_EXAMPLES = [
    {"detected_language": "en",      "user_message": "hello there",                         "intent": "greeting"},
    {"detected_language": "ar",      "user_message": "شكرا جدا",                            "intent": "gratitude"},
    {"detected_language": "ar",      "user_message": "انا حاسس بقلق طول الوقت",             "intent": "asking_mental_health_question"},
    {"detected_language": "en",      "user_message": "what's the weather like",              "intent": "out_of_scope"},
    {"detected_language": "en",      "user_message": "thanks, bye",                          "intent": "gratitude"},
    {"detected_language": "en",      "user_message": "I want to end my life, it's too much", "intent": "crisis"},
    {"detected_language": "unknown", "user_message": "عايز انتحر وارتاح من الدنيا",         "intent": "crisis"},
    {"detected_language": "en",      "user_message": "I am going to hurt someone else",      "intent": "crisis"},
]

_INSTRUCTIONS = instructions = """
ROLE
-----
You are an intent classifier for a mental health support chatbot.

TASK
-----
Classify the user's message into exactly one of these intents:
greeting | goodbye | gratitude | asking_mental_health_question | out_of_scope | crisis

RULES
-----
- Return ONLY a JSON object, no explanation.
- Output must be valid JSON.
- Do NOT add extra text or formatting.
- If unsure, prefer asking_mental_health_question over out_of_scope.
- Suicide, self-harm, or intent/thoughts of hurting oneself or anyone else MUST be classified as crisis.
- Prioritize: crisis > asking_mental_health_question > gratitude > greeting > goodbye > out_of_scope
- If detected_language was unkown, Answer with the same language as the query language.

OUTPUT FORMAT
-----
{{"intent": "<label>"}}
"""


_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Detected language: {detected_language}\nUser: \"{user_message}\""),
    ("ai",    '{{"intent": "{intent}"}}'),
])

_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt,
    examples=_EXAMPLES,
)

# ✅ This is the only thing other files should import
intent_system_prompt = ChatPromptTemplate.from_messages([
    ("system", _INSTRUCTIONS),
    _few_shot,
    ("human", "Detected language: {detected_language}\nRecent context:\n{recent_context}\nUser: \"{user_message}\""),
])