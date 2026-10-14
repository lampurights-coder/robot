import json
import requests

QUESTION = "برای بی‌حسی لب بعد از دندون عقل، لیزر کم‌توان کمک می‌کنه؟"

r = requests.post(
    "http://127.0.0.1:7860/ask",
    json={"question": QUESTION},
)
data = r.json()
data["your_question"] = QUESTION  # what you asked
# matched_question = FAQ card question that was retrieved (can differ)

print(data["answer"])

with open("answer.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("wrote answer.json")
