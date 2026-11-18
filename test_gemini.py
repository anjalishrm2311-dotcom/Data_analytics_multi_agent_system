"""
Run this OUTSIDE Streamlit to find out which Gemini model actually works for your key.

    python test_gemini.py

It lists the models your key can use, then sends a tiny question to each
Flash-type model and prints how long it took, or the exact error.
"""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("Key found:", bool(key))
if not key:
    raise SystemExit("No GEMINI_API_KEY. Check your .env file.")

client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20_000))

names = []
try:
    for m in client.models.list():
        name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions and "gemini" in name:
            names.append(name)
except Exception as e:
    print("Could not list models:", e)

print("\nModels your key can use for text:")
for n in names:
    print("  ", n)

candidates = [n for n in names if "flash" in n and "image" not in n and "tts" not in n][:8]
print("\nTesting up to 8 flash models (20 second limit each):\n")

for n in candidates:
    start = time.time()
    try:
        r = client.models.generate_content(
            model=n,
            contents="Reply with the single word: ok",
            config=types.GenerateContentConfig(max_output_tokens=200),
        )
        print(f"OK    {n:40s} {time.time() - start:5.1f}s  -> {(r.text or '').strip()!r}")
    except Exception as e:
        print(f"FAIL  {n:40s} {time.time() - start:5.1f}s  -> {str(e)[:140]}")