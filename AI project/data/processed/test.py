import json
chunks = json.load(open("acg_chunks.json"))
hits = [c["chunk_id"] for c in chunks if "on 11/25/2024" in c["text"] or "abggQZXdtwnfKZBYtws" in c["text"]]
print(len(hits), hits)