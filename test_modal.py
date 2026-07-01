import httpx, time, json

URL = "https://adhityaganesh49--llm-benchmark-vllmserver-serve.modal.run/v1/chat/completions"
payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
    "max_tokens": 100,
    "stream": True,
}

print("Sending request (cold start expected ~2-3 min)...")
t_start = time.perf_counter()
t_first = None
token_count = 0

with httpx.Client(timeout=300) as client:
    with client.stream("POST", URL, json=payload) as resp:
        print(f"HTTP {resp.status_code}  [{time.perf_counter()-t_start:.2f}s]")
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                token = json.loads(data)["choices"][0]["delta"].get("content", "")
                if token:
                    if t_first is None:
                        t_first = time.perf_counter()
                        print(f"\n*** FIRST TOKEN: {t_first - t_start:.2f}s ***\n")
                    token_count += 1
                    print(token, end="", flush=True)
            except Exception:
                pass

t_end = time.perf_counter()
print("\n\n--- Results ---")
if t_first:
    print(f"TTFT:  {t_first - t_start:.2f}s")
    print(f"Total: {t_end - t_start:.2f}s")
    print(f"Tokens: {token_count}")
    if token_count > 1:
        print(f"TPOT:  {(t_end - t_first) / (token_count - 1) * 1000:.1f}ms/token")
else:
    print(f"No tokens. Status: {resp.status_code}")
