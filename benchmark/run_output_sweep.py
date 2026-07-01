"""Output-length sweep: measures how TPOT and total latency scale with output tokens.

Hypothesis: total latency grows linearly with output length (decode is memory-bandwidth-bound),
while TPOT stays roughly constant (per-token cost is fixed).
"""

import argparse
import json
import time
from pathlib import Path
from harness import stream_request, DEFAULT_PROMPT

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_LENGTHS = [25, 50, 100, 200, 400, 800]


def run(platform: str, n: int):
    results = []
    for max_tokens in OUTPUT_LENGTHS:
        for i in range(n):
            print(f"  max_tokens={max_tokens} run {i+1}/{n} ... ", end="", flush=True)
            r = stream_request(platform, prompt=DEFAULT_PROMPT, max_tokens=max_tokens)
            row = {
                "max_tokens": max_tokens,
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
                print(f"TTFT={r.ttft:.3f}s  TPOT={r.tpot:.4f}s  "
                      f"total={r.total_latency:.2f}s  tokens={r.output_tokens}")

    out = RAW_DIR / f"output_sweep_{platform}_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["modal", "simplismart"])
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    run(args.platform, args.n)
