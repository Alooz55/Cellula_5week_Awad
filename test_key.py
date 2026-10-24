"""Quick check of the OpenRouter key, independent of LangChain. Run: python test_key.py"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
print("key length:", len(key), "| prefix:", key[:8], "| non-ascii chars:", any(ord(c) > 127 for c in key))

headers = {"Authorization": f"Bearer {key}"}

r = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=20)
print("1) key check ->", r.status_code, r.text[:300])

r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json={"model": "openrouter/free", "messages": [{"role": "user", "content": "say hi"}]},
    timeout=60,
)
print("2) chat call ->", r.status_code, r.text[:300])
