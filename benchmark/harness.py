"""Shared benchmark harness: direct SSE streaming (no OpenAI SDK) for reliability."""

import time
import json
import os
import httpx
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

PLATFORMS = {
    "modal": os.environ.get("MODAL_ENDPOINT_URL", ""),
    "simplismart": os.environ.get("SIMPLISMART_ENDPOINT_URL", ""),
}

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
SIMPLISMART_TOKEN = os.environ.get("SIMPLISMART_PG_TOKEN", "")
SIMPLISMART_DEPLOYMENT_ID = os.environ.get("SIMPLISMART_DEPLOYMENT_ID", "")
SIMPLISMART_MODEL_NAME = os.environ.get("SIMPLISMART_MODEL_NAME", "qwen2d5")
DEFAULT_PROMPT = (
    "Explain the difference between latency and throughput in distributed systems. "
    "Be concise but thorough."
)
DEFAULT_MAX_TOKENS = 200


@dataclass
class TokenEvent:
    token: str
    timestamp: float


@dataclass
class RequestResult:
    platform: str
    prompt_tokens: int
    request_start: float
    first_token_time: float | None = None
    tokens: list[TokenEvent] = field(default_factory=list)
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.request_start

    @property
    def output_tokens(self) -> int:
        return len(self.tokens)

    @property
    def total_latency(self) -> float | None:
        if not self.tokens:
            return None
        return self.tokens[-1].timestamp - self.request_start

    @property
    def tpot(self) -> float | None:
        if len(self.tokens) < 2 or self.first_token_time is None:
            return None
        decode_time = self.tokens[-1].timestamp - self.first_token_time
        return decode_time / (len(self.tokens) - 1)


def stream_request(
    platform: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str = "placeholder",
) -> RequestResult:
    base_url = PLATFORMS[platform].rstrip("/")
    if not base_url:
        raise ValueError(f"No URL set for '{platform}'")

    if platform == "simplismart":
        url = f"{base_url}/chat/completions"
        model_name = SIMPLISMART_MODEL_NAME
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SIMPLISMART_TOKEN}",
            "id": SIMPLISMART_DEPLOYMENT_ID,
        }
    else:
        # Modal/OpenAI-compatible: strip /v1 suffix if present, then re-append
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        url = f"{base_url}/v1/chat/completions"
        model_name = MODEL_ID
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    result = RequestResult(platform=platform, prompt_tokens=0,
                           request_start=time.perf_counter())
    try:
        with httpx.Client(timeout=300) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    result.error = f"HTTP {resp.status_code}"
                    return result
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            ts = time.perf_counter()
                            if result.first_token_time is None:
                                result.first_token_time = ts
                            result.tokens.append(TokenEvent(token=delta, timestamp=ts))
                    except Exception:
                        pass
    except Exception as e:
        result.error = str(e)

    return result
