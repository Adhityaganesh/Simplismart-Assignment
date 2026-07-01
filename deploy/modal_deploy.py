import os
import modal

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

GPU_TYPE = "L4"
SCALEDOWN_WINDOW = 120
MAX_MODEL_LEN = 4096
VLLM_PORT = 8000

APP_NAME = "llm-benchmark"
VOLUME_NAME = "llm-benchmark-weights"
WEIGHTS_DIR = "/weights"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.4.3", "huggingface_hub>=0.23.0", "aiohttp")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


@app.function(
    image=vllm_image,
    volumes={WEIGHTS_DIR: volume},
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_ID,
        local_dir=f"{WEIGHTS_DIR}/{MODEL_ID}",
        ignore_patterns=["*.pt", "*.bin"],
    )
    volume.commit()
    print("Download complete.")


@app.cls(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={WEIGHTS_DIR: volume},
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=600,
)
@modal.concurrent(max_inputs=32)
class VLLMServer:
    @modal.enter()
    def start_server(self):
        import subprocess, time, httpx

        self._proc = subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", f"{WEIGHTS_DIR}/{MODEL_ID}",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--dtype", "half",
            "--served-model-name", MODEL_ID,
            "--enforce-eager",
        ])

        client = httpx.Client()
        for _ in range(180):
            try:
                if client.get(f"http://localhost:{VLLM_PORT}/health").status_code == 200:
                    print("vLLM ready.")
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("vLLM did not start within 180s")

    @modal.exit()
    def stop_server(self):
        self._proc.terminate()

    @modal.asgi_app()
    def serve(self):
        import aiohttp
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse

        proxy = FastAPI(redirect_slashes=False)

        HOP_BY_HOP = frozenset([
            "transfer-encoding", "connection", "keep-alive",
            "te", "trailers", "upgrade", "content-length",
        ])

        @proxy.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"])
        async def forward(path: str, request: Request):
            url = f"http://localhost:{VLLM_PORT}/{path}"
            body = await request.body()
            headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in ("host", "content-length")}

            # aiohttp avoids the anyio backend conflict that httpx.AsyncClient hits
            # inside Modal's ASGI runtime, which caused ReadError mid-stream.
            session = aiohttp.ClientSession()
            resp = await session.request(
                request.method, url,
                data=body, headers=headers,
                params=dict(request.query_params),
                timeout=aiohttp.ClientTimeout(total=300),
            )

            safe_headers = {k: v for k, v in resp.headers.items()
                            if k.lower() not in HOP_BY_HOP}

            async def _stream():
                try:
                    async for chunk in resp.content.iter_chunked(4096):
                        yield chunk
                finally:
                    await resp.release()
                    await session.close()

            return StreamingResponse(
                _stream(),
                status_code=resp.status,
                headers=safe_headers,
                media_type=resp.content_type,
            )

        return proxy


@app.local_entrypoint()
def main():
    print(f"GPU: {GPU_TYPE} (~$0.80/hr L4)")
    print("Downloading weights (cached if already done)...")
    download_model.remote()
    print("Done. Run: modal deploy deploy/modal_deploy.py")
