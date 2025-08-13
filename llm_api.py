import openai
import os
import time
import json

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
        "what_to_do": "[Test] Stretch your shoulders and chest.",
        "summary": "[Test] According to the data, you keep your back straight more than 80% of the time. According to the data, you keep your back straight more than 80% of the time.  According to the data, you keep your back straight more than 80% of the time.  According to the data, you keep your back straight more than 80% of the time. That's great!According to the data, you keep your back straight more than 80% of the time. That's great!According to the data, you keep your back straight more than 80% of the time. That's great!According to the data, you keep your back straight more than 80% of the time. That's great!According to the data, you keep your back straight more than 80% of the time. That's great!",
        "recommendation": "[Test]  I recommend developing a habit of maintaining your current posture."
    }

def real_get_analysis_data():
    try:
        # 예: latest_data = get_data_from_db()
        #     prompt = make_analysis_prompt(latest_data)
        #     response = openai.ChatCompletion.create(...)
        #     result = json.loads(response.choices[0].message['content'])
        #     return result
        return mock_get_analysis_data()
    except Exception as e:
        return {"error": str(e)}

# ----------------- Pipeline 2: Chat -----------------

def get_chat_response(user_message: str, chat_history: list):
    print("--- Chat Pipeline Triggered ---")
    if USE_REAL_LLM:
        return real_get_chat_response(user_message, chat_history)
    else:
        return mock_get_chat_response(user_message, chat_history)

def mock_get_chat_response(user_message: str, chat_history: list):
    time.sleep(1)
    return {"answer": f"Hello"}

def real_get_chat_response(user_message: str, chat_history: list):
    try:
        # messages = chat_history + [{"role": "user", "content": user_message}]
        # response = openai.ChatCompletion.create(...)
        # answer = response.choices[0].message['content']
        # return {"answer": answer}
        return mock_get_chat_response(user_message, chat_history)
    except Exception as e:
        return {"error": str(e)}