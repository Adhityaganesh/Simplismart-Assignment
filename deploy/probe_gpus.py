"""Probe which GPU IDs have available quota by attempting a dummy compile."""
import os, json
from simplismart.http import HTTPClient
from simplismart.config import SimplismartConfig
from simplismart.models.model_repo import ModelRepoCompileCreate, ModelRepoCompileAvatar

TOKEN = os.environ["SIMPLISMART_PG_TOKEN"]
ORG_ID = "25fdc8cb-b43e-42ad-8254-28e96a50e801"

http = HTTPClient(SimplismartConfig(pg_token=TOKEN))

GPU_IDS = [
    "nvidia-h100", "nvidia-a100", "nvidia-a10g", "nvidia-a10",
    "nvidia-l4", "nvidia-t4", "nvidia-v100",
    "h100", "a100", "a10g", "a10", "l4", "t4", "v100",
    "A100", "A10G", "A10", "L4", "T4", "H100",
]

print("Checking GPU availability via compile probe...\n")
for gpu_id in GPU_IDS:
    try:
        payload = ModelRepoCompileCreate(
            name=f"probe-{gpu_id}",
            avatar=ModelRepoCompileAvatar(image_url="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
            source_type="huggingface",
            source_url="Qwen/Qwen2.5-7B-Instruct",
            model_class="Qwen2ForCausalLM",
            accelerator_type=gpu_id,
            org=ORG_ID,
            use_simplismart_infrastructure=True,
        )
        body = payload.to_payload()
        body["use_simplismart_infrastructure"] = True
        result = http.request("POST", "/models/client/model-repos/", json=body)
        repo_uuid = result.get("uuid") or result.get("data", {}).get("uuid", "?")
        print(f"  {gpu_id}: ACCEPTED (uuid={repo_uuid}) — deleting probe...")
        try:
            http.request("DELETE", "/models/client/model-repos/", params={"model_id": repo_uuid})
        except Exception:
            pass
    except Exception as e:
        print(f"  {gpu_id}: {e}")
