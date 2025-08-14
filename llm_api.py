import openai
import time
import json
from langchain_core.messages import HumanMessage, AIMessage


USE_REAL_LLM = False 

# ----------------- Pipeline 1: Analysis -----------------

def get_analysis_data():
    print("--- Analysis Pipeline Triggered ---")
    if USE_REAL_LLM:
        return real_get_analysis_data()
    else:
        return mock_get_analysis_data()

def mock_get_analysis_data():
    time.sleep(1.5)
    return {
        "Do this now": "[Test] Stretch your shoulders and chest.",
        "Why this matters": "[Test] You've maintained an upright posture for 85% of the last session. Your key habit to guard is to avoid leaning forward when typing.",
        "Summary and Habit Guard": "[Test] Consistently holding an upright posture reduces strain on your lower back and prevents long-term spine issues. It also improves focus and energy levels.",
        "Short Term Plan":  "[Test] For the next hour, focus on keeping your shoulders relaxed and away from your ears. Set a brief reminder to check in 15 minutes."
    }

def real_get_analysis_data():
    pass

# ----------------- Pipeline 2: Chat -----------------

def get_chat_response(user_message: str, chat_history: list):
    print("--- Chat Pipeline Triggered ---")
    if USE_REAL_LLM:
        return real_get_chat_response(user_message, chat_history)
    else:
        return mock_get_chat_response(user_message, chat_history)

def mock_get_chat_response(user_message: str, chat_history: list):
    time.sleep(1)
    return AIMessage(content=f"[채팅 테스트] '{user_message}'라고 말씀하셨네요! 반갑습니다.")

def real_get_chat_response(user_message: str, chat_history: list):
    pass