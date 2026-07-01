"""Input-length sweep: measures how TTFT scales with prompt token count (prefill).

Hypothesis: TTFT grows linearly with input length (prefill is compute-bound).
"""

import argparse
import json
import time
from pathlib import Path
from harness import stream_request

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Prompts of increasing length (roughly 64 / 128 / 256 / 512 / 1024 tokens)
PROMPTS = {
    64:   "Explain what a neural network is. Be concise.",
    128:  ("Explain the difference between supervised and unsupervised learning in machine learning. "
           "Give one concrete example of each, and describe when you would choose one over the other. "
           "Be clear and concise."),
    256:  ("You are a distributed systems expert. A junior engineer asks: "
           "'We have a microservices architecture. Some services are written in Python, some in Go. "
           "They communicate over REST. We are seeing high tail latency at p99. "
           "What are the most common root causes of high p99 latency in microservice architectures, "
           "and what are the standard techniques to diagnose and fix them? "
           "Please be specific about tools and approaches.' Answer thoroughly."),
    512:  ("You are a senior software engineer being asked to review a system design for a real-time "
           "analytics platform that must ingest one million events per second, store them durably, "
           "and serve sub-second queries over a 30-day rolling window. "
           "The candidate has proposed: Kafka for ingestion, Apache Flink for stream processing, "
           "Apache Druid for OLAP storage, Redis for hot-path caching, and a React dashboard. "
           "Evaluate this design in depth. Cover: (1) whether each technology is the right choice, "
           "(2) the likely failure modes at 1M events/sec, "
           "(3) the hardest operational challenges, "
           "(4) what you would change and why, "
           "(5) how you would test and validate this system before production. "
           "Be rigorous, specific, and opinionated. Avoid generic advice."),
    1024: ("You are the principal architect at a company building a multi-tenant SaaS platform for "
           "financial services firms. The platform must meet SOC 2 Type II and PCI DSS Level 1 "
           "requirements. It serves 500 enterprise clients, each with up to 10,000 users. "
           "The platform processes payment transactions, generates compliance reports, and provides "
           "real-time fraud detection. Current stack: Python/FastAPI microservices, PostgreSQL (RDS), "
           "Redis, Celery for async tasks, deployed on AWS EKS. "
           "The CTO has asked you to design the next-generation architecture that will: "
           "(a) support 10x current transaction volume without re-architecting again, "
           "(b) achieve 99.99% uptime SLA, "
           "(c) enable zero-downtime deployments, "
           "(d) provide full audit trails for compliance, "
           "(e) support multi-region active-active deployment in US, EU, and APAC. "
           "Write a complete architectural proposal. Include: "
           "the high-level architecture diagram described in words, "
           "database strategy including sharding and replication, "
           "the event sourcing / CQRS pattern for audit trails, "
           "the approach to multi-region consistency (CAP theorem trade-offs), "
           "the CI/CD and deployment strategy, "
           "the observability stack, "
           "the estimated infrastructure cost breakdown, "
           "and a 12-month phased migration plan from the current architecture. "
           "Be specific about technology choices with justification for each."),
}


def run(platform: str, n: int):
    results = []
    for target_len, prompt in PROMPTS.items():
        for i in range(n):
            print(f"  input_len~{target_len} run {i+1}/{n} ... ", end="", flush=True)
            r = stream_request(platform, prompt=prompt, max_tokens=50)
            row = {
                "target_input_tokens": target_len,
                "actual_prompt_chars": len(prompt),
                "run": i,
                "platform": platform,
                "ttft_s": r.ttft,
                "tpot_s": r.tpot,
                "output_tokens": r.output_tokens,
                "error": r.error,
            }
            results.append(row)
            if r.error or r.ttft is None:
                print(f"ERROR: {r.error}")
            else:
                print(f"TTFT={r.ttft:.3f}s  tokens_out={r.output_tokens}")

    out = RAW_DIR / f"input_sweep_{platform}_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["modal", "simplismart"])
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    run(args.platform, args.n)
