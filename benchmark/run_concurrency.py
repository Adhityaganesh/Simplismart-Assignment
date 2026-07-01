"""Concurrency sweep: fires N parallel requests and measures throughput."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from harness import stream_request, DEFAULT_PROMPT, DEFAULT_MAX_TOKENS

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


async def _one(platform, prompt, max_tokens, idx):
    loop = asyncio.get_event_loop()
    r = await loop.run_in_executor(
        None, lambda: stream_request(platform, prompt=prompt, max_tokens=max_tokens)
    )
    return {
        "worker": idx,
        "platform": platform,
        "ttft_s": r.ttft,
        "tpot_s": r.tpot,
        "total_latency_s": r.total_latency,
        "output_tokens": r.output_tokens,
        "error": r.error,
    }


async def sweep(platform, concurrency_levels, prompt, max_tokens):
    all_results = {}
    for c in concurrency_levels:
        print(f"  concurrency={c} ...", flush=True)
        t0 = time.perf_counter()
        rows = await asyncio.gather(*[_one(platform, prompt, max_tokens, i) for i in range(c)])
        wall = time.perf_counter() - t0
        total_tokens = sum(r["output_tokens"] for r in rows if r["output_tokens"])
        print(f"    wall={wall:.2f}s  total_tokens={total_tokens}  throughput={total_tokens/wall:.1f} tok/s")
        all_results[str(c)] = {"wall_s": wall, "requests": list(rows)}

    out = RAW_DIR / f"concurrency_{platform}_{int(time.time())}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["modal", "simplismart"])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    asyncio.run(sweep(args.platform, args.concurrency, args.prompt, args.max_tokens))
