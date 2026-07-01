"""Single-stream latency benchmark: measures TTFT and TPOT over N runs."""

import argparse
import json
import time
import os
from pathlib import Path
from harness import stream_request, DEFAULT_PROMPT, DEFAULT_MAX_TOKENS

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def run(platform: str, n: int, prompt: str, max_tokens: int):
    results = []
    for i in range(n):
        print(f"  run {i+1}/{n} ... ", end="", flush=True)
        r = stream_request(platform, prompt=prompt, max_tokens=max_tokens)
        row = {
            "run": i,
            "platform": platform,
            "ttft_s": r.ttft,
            "tpot_s": r.tpot,
            "total_latency_s": r.total_latency,
            "output_tokens": r.output_tokens,
            "error": r.error,
        }
        results.append(row)
        if r.error or r.ttft is None:
            print(f"ERROR: {r.error}")
        else:
            print(f"TTFT={r.ttft:.3f}s  TPOT={r.tpot:.4f}s  tokens={r.output_tokens}")

    out = RAW_DIR / f"latency_{platform}_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["modal", "simplismart"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    run(args.platform, args.n, args.prompt, args.max_tokens)
