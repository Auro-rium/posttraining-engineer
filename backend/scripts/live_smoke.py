"""Non-destructive AWS smoke test for Strands, Bedrock, and S3.

The script creates no AWS resources. It writes one temporary object to an
existing bucket, verifies its bytes and SHA-256 metadata, then deletes it.
SageMaker, DynamoDB, ECS, and AgentCore are intentionally not invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from typing import Any

import boto3
from strands import Agent
from strands.models import BedrockModel


class SmokeFailure(RuntimeError):
    """Raised when a live smoke-test contract fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.getenv("S3_ARTIFACT_BUCKET"), required=False)
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument(
        "--model-id", default=os.getenv("STRANDS_MODEL", "nvidia.nemotron-super-3-120b")
    )
    return parser.parse_args()


def invoke_strands(region: str, model_id: str) -> str:
    model = BedrockModel(model_id=model_id, region_name=region)
    agent = Agent(model=model, system_prompt="Reply with exactly READY and nothing else.")
    response = str(agent("health check"))
    if "READY" not in response.upper():
        raise SmokeFailure(f"Bedrock response did not contain READY: {response[:100]!r}")
    return response


def s3_round_trip(bucket: str, region: str) -> dict[str, Any]:
    payload = b"autonomous-post-training-live-smoke\n"
    digest = hashlib.sha256(payload).hexdigest()
    key = f"_smoke/{uuid.uuid4().hex}.bin"
    client = boto3.client("s3", region_name=region)
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            Metadata={"sha256": digest, "purpose": "live-smoke"},
            ContentType="application/octet-stream",
        )
        result = client.get_object(Bucket=bucket, Key=key)
        received = result["Body"].read()
        if received != payload:
            raise SmokeFailure("S3 round-trip bytes did not match")
        if result.get("Metadata", {}).get("sha256") != digest:
            raise SmokeFailure("S3 SHA-256 metadata did not match")
        return {"bucket": bucket, "key": key, "sha256": digest, "deleted": True}
    finally:
        client.delete_object(Bucket=bucket, Key=key)


def main() -> int:
    args = parse_args()
    if not args.bucket:
        raise SystemExit("Pass --bucket or set S3_ARTIFACT_BUCKET")

    identity = boto3.client("sts", region_name=args.region).get_caller_identity()
    response = invoke_strands(args.region, args.model_id)
    artifact = s3_round_trip(args.bucket, args.region)
    print(
        {
            "status": "passed",
            "account": identity.get("Account"),
            "region": args.region,
            "model_id": args.model_id,
            "bedrock_response": response[:100],
            "s3_artifact": artifact,
            "resources_created": [],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
