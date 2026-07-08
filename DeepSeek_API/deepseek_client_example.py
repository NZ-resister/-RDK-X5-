import json
import os
import urllib.request


def call_deepseek(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY first.")

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a concise language-learning assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 180,
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
    result = json.loads(body)
    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    prompt = (
        "Return JSON only. Generate one beginner-friendly English sentence "
        "using the word apple and provide a Chinese translation. "
        "Fields: example, translation."
    )
    print(call_deepseek(prompt))

