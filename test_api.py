"""Quick test to see what Groq returns."""
import os
import json
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not set")
    exit(1)

client = Groq(api_key=api_key)

# Simple test
response = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[
        {
            "role": "system",
            "content": "You are a JSON machine. Output ONLY valid JSON, nothing else."
        },
        {
            "role": "user", 
            "content": """Output JSON with these fields: name, age, city. That's it. Only JSON.

{"name": "John", "age": 30, "city": "NYC"}"""
        }
    ],
    temperature=0.2,
    max_tokens=100,
)

text = response.choices[0].message.content.strip()
print(f"Raw response:\n{text}\n")

# Try to parse
try:
    data = json.loads(text)
    print(f"✅ Valid JSON: {data}")
except Exception as e:
    print(f"❌ Parse error: {e}")
