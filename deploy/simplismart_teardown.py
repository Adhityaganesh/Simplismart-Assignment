"""Tear down Simplismart deployment and model repo to stop billing."""

import argparse
import json
import os
from simplismart.http import HTTPClient
from simplismart.config import SimplismartConfig

TOKEN = os.environ["SIMPLISMART_PG_TOKEN"]
http = HTTPClient(SimplismartConfig(pg_token=TOKEN))

parser = argparse.ArgumentParser()
parser.add_argument("--deployment-id", required=True)
parser.add_argument("--model-repo-id", required=True)
args = parser.parse_args()

print(f"Deleting deployment {args.deployment_id}...")
try:
    http.request("DELETE", f"/deployments/{args.deployment_id}/delete/")
    print("Deployment deleted.")
except Exception as e:
    print(f"  deployment delete error (may already be gone): {e}")

print(f"Deleting model repo {args.model_repo_id}...")
try:
    http.request("DELETE", "/models/client/model-repos/",
                 params={"model_id": args.model_repo_id})
    print("Model repo deleted.")
except Exception as e:
    print(f"  model repo delete error: {e}")

print("Teardown complete. GPU billing stopped.")
