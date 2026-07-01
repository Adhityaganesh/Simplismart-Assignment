# LLM Inference Benchmark: Simplismart vs Modal

Benchmarks TTFT, TPOT, throughput, and cost/token for the same model served on
Simplismart and Modal, using their CLI/SDK (no web UI).

## Assumptions

1. **Model**: `Qwen/Qwen2.5-7B-Instruct` (**assumption**: substituted for
   `meta-llama/Llama-3.1-8B-Instruct` because Llama 3.1 is gated on HuggingFace
   and requires manual approval). Qwen2.5-7B-Instruct is ungated, similarly sized
   (~7B parameters, fits in ~14 GB fp16), instruction-tuned, and openly licensed.
   All benchmark results and cost figures in the report apply to this model.
   To switch back to Llama once access is granted, set `MODEL_ID=meta-llama/Llama-3.1-8B-Instruct`
   and `HF_TOKEN=<your_token>` in `.env`.

2. **GPU**: L4 (24 GB VRAM) on Modal. Llama 3.1 8B in fp16 requires ~16 GB;
   the L4 is the cheapest GPU that fits with headroom. An A10G is the fallback
   if the L4 is unavailable.

3. **Scale-to-zero**: `scaledown_window=120s` (2 minutes) to minimize idle
   billing. Cold-start latency is explicitly measured as a benchmark metric.

4. **Serving**: vLLM in OpenAI-compatible mode (`/v1/chat/completions`) with
   streaming enabled. The same client code hits both platforms.

5. **Benchmark workload**: fixed prompt (~50 tokens) requesting 200-token
   completions; concurrency sweep: 1, 2, 4, 8, 16 parallel requests;
   cold-start measured after intentional idle period.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in HF_TOKEN, MODAL_API_KEY, SIMPLISMART_API_KEY
```

## Deployment

```bash
# Modal
modal deploy deploy/modal_deploy.py

# Simplismart
python deploy/simplismart_deploy.py
```

## Running Benchmarks

```bash
# Single platform latency sweep
python benchmark/run_latency.py --platform modal

# Concurrency sweep
python benchmark/run_concurrency.py --platform modal --concurrency 1 2 4 8 16

# Cold-start test
python benchmark/run_coldstart.py --platform modal

# Analyze + chart
python benchmark/analyze.py
```

## Cost Model

- Modal L4: ~$0.80/hr; A10G: ~$1.10/hr
- Simplismart: per-token pricing (TBD from their dashboard after deployment)
- Cost/token computed as `(gpu_seconds / 3600) * hourly_rate / output_tokens`
# Simplismart-Assignment
