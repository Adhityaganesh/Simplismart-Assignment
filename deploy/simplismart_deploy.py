"""
Simplismart deployment script for Qwen/Qwen2.5-7B-Instruct.
Steps: compile model → poll until ready → deploy → poll until healthy → print endpoint.
"""

import json
import os
import time

from simplismart.http import HTTPClient
from simplismart.config import SimplismartConfig
from simplismart.models.model_repo import ModelRepoCompileCreate, ModelRepoCompileAvatar
from simplismart.models.deployment import DeploymentCreate

TOKEN = os.environ["SIMPLISMART_PG_TOKEN"]
ORG_ID = "25fdc8cb-b43e-42ad-8254-28e96a50e801"  # decoded from JWT org_uuid
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GPU_ID = "nvidia-a10"
DEPLOY_NAME = "qwen25-7b-benchmark"

http = HTTPClient(SimplismartConfig(pg_token=TOKEN))


def compile_model() -> str:
    """Submit compilation job. Returns model_repo UUID."""
    print(f"Compiling {MODEL_ID} on {GPU_ID}...")
    payload = ModelRepoCompileCreate(
        name="Qwen2.5-7B-Instruct-Benchmark",
        avatar=ModelRepoCompileAvatar(
            image_url="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
        ),
        source_type="huggingface",
        source_url=MODEL_ID,
        model_class="Qwen2ForCausalLM",
        accelerator_type=GPU_ID,
        org=ORG_ID,
        use_simplismart_infrastructure=True,
    )
    body = payload.to_payload()
    body["use_simplismart_infrastructure"] = True
    result = http.request("POST", "/models/client/model-repos/", json=body)
    print(json.dumps(result, indent=2))
    uuid = result.get("uuid") or result.get("data", {}).get("uuid")
    if not uuid:
        raise RuntimeError(f"No UUID in compile response: {result}")
    print(f"\nModel repo UUID: {uuid}")
    return uuid


def poll_compile(model_repo_uuid: str, timeout: int = 1800) -> None:
    """Poll until model repo status is SUCCESS."""
    print("Polling compile status...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = http.request("GET", "/models/client/model-repos/",
                              params={"model_id": model_repo_uuid, "count": 1})
        items = result.get("results", [result] if isinstance(result, dict) else [])
        for item in items:
            if item.get("uuid") == model_repo_uuid:
                status = item.get("status", "UNKNOWN")
                print(f"  compile status: {status}  [{int(time.time())}]")
                if status == "SUCCESS":
                    print("Compile complete.")
                    return
                if status in ("FAILED", "ERROR"):
                    raise RuntimeError(f"Compile failed: {item}")
        time.sleep(30)
    raise TimeoutError("Compile timed out after 30 minutes")


def deploy(model_repo_uuid: str) -> str:
    """Create deployment. Returns deployment UUID."""
    print(f"\nDeploying {DEPLOY_NAME}...")
    payload = DeploymentCreate(
        model_repo=model_repo_uuid,
        org=ORG_ID,
        gpu_id=GPU_ID,
        name=DEPLOY_NAME,
        min_pod_replicas=1,
        max_pod_replicas=1,
        scale_to_zero_enabled=True,
    )
    result = http.request("POST", "/deployments/private/deploy-model/",
                          json=payload.to_payload())
    print(json.dumps(result, indent=2))
    dep_uuid = (result.get("uuid") or
                result.get("id") or
                result.get("data", {}).get("uuid"))
    if not dep_uuid:
        raise RuntimeError(f"No UUID in deploy response: {result}")
    print(f"\nDeployment UUID: {dep_uuid}")
    return dep_uuid


def poll_deployment(dep_uuid: str, timeout: int = 900) -> str:
    """Poll until deployment is healthy. Returns inference endpoint URL."""
    print("Polling deployment health...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health = http.request(
                "GET", f"/deployments/model-deployment/fetch-status/{dep_uuid}/")
            status = health.get("data") or health.get("status") or str(health)
            print(f"  deployment status: {status}  [{int(time.time())}]")

            if isinstance(status, str) and "healthy" in status.lower():
                # Try to extract endpoint URL from deployment details
                detail = http.request(
                    "GET", f"/deployments/model-deployments/{dep_uuid}/")
                print("Deployment detail:")
                print(json.dumps(detail, indent=2))
                endpoint = extract_endpoint(detail)
                return endpoint
        except Exception as e:
            print(f"  health check error: {e}")
        time.sleep(20)
    raise TimeoutError("Deployment timed out after 15 minutes")


def extract_endpoint(detail: dict) -> str:
    """Pull inference endpoint URL out of deployment detail response."""
    # Try common field names
    for key in ("endpoint", "inference_endpoint", "url", "endpoint_url",
                "base_url", "service_url", "host"):
        val = detail.get(key)
        if val and isinstance(val, str) and val.startswith("http"):
            return val
    # Recurse into nested dicts
    for val in detail.values():
        if isinstance(val, dict):
            url = extract_endpoint(val)
            if url:
                return url
    # Fallback: construct from deployment UUID pattern
    print("WARNING: could not find endpoint URL in response — using constructed URL")
    dep_uuid = detail.get("uuid") or detail.get("id", "UNKNOWN")
    return f"https://api.app.simplismart.ai/deployments/{dep_uuid}"


def update_env(endpoint_url: str) -> None:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(env_path) as f:
        content = f.read()
    key = "SIMPLISMART_ENDPOINT_URL"
    line = f"{key}={endpoint_url}"
    if key in content:
        lines = [line if l.startswith(key) else l for l in content.splitlines()]
        content = "\n".join(lines) + "\n"
    else:
        content = content.rstrip("\n") + f"\n{line}\n"
    with open(env_path, "w") as f:
        f.write(content)
    print(f"\n.env updated: {key}={endpoint_url}")


if __name__ == "__main__":
    model_repo_uuid = compile_model()
    poll_compile(model_repo_uuid)
    dep_uuid = deploy(model_repo_uuid)
    endpoint = poll_deployment(dep_uuid)
    print(f"\n=== ENDPOINT: {endpoint} ===")
    update_env(endpoint)
    print("\nDeployment complete. Run benchmarks, then tear down with:")
    print(f"  python deploy/simplismart_teardown.py --deployment-id {dep_uuid} --model-repo-id {model_repo_uuid}")
