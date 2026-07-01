"""Cold-start benchmark: waits for scale-to-zero then measures first-request latency."""

import argparse
import json
import time
from pathlib import Path
from harness import stream_request, DEFAULT_PROMPT, DEFAULT_MAX_TOKENS

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Wait longer than the scaledown_window (120 s) to ensure the GPU is gone
IDLE_WAIT_S = 150


def run(platform: str, prompt: str, max_tokens: int):
    print(f"Waiting {IDLE_WAIT_S}s for scale-to-zero to kick in...")
    time.sleep(IDLE_WAIT_S)

    print("Firing cold-start request...")
    t0 = time.perf_counter()
    r = stream_request(platform, prompt=prompt, max_tokens=max_tokens)
    cold_wall = time.perf_counter() - t0

    result = {
        "platform": platform,
        "cold_wall_s": cold_wall,
        "ttft_s": r.ttft,
        "tpot_s": r.tpot,
        "output_tokens": r.output_tokens,
        "error": r.error,
    }
    print(f"Cold-start wall={cold_wall:.2f}s  TTFT={r.ttft:.3f}s")

    out = RAW_DIR / f"coldstart_{platform}_{int(time.time())}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["modal", "simplismart"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    run(args.platform, args.prompt, args.max_tokens)
