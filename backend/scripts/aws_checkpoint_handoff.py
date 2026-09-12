"""Stage the pinned FunctionGemma snapshot from Hugging Face inside AWS.

This is a single-purpose handoff worker.  It accepts its Hugging Face token
only through an environment variable, downloads an explicit immutable commit
to a temporary directory, then delegates validation and versioned S3 upload to
``stage_functiongemma_checkpoint``.  It never prints the token, model output,
or local checkpoint path.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from typing import Any

from stage_functiongemma_checkpoint import (
    TARGET_MODEL_ID,
    CheckpointStagingError,
    stage_checkpoint,
    validate_immutable_revision,
)


class CheckpointHandoffError(RuntimeError):
    """The isolated remote checkpoint handoff could not produce evidence."""


def stage_from_huggingface(
    *,
    bucket: str,
    revision: str,
    prefix: str,
    region: str | None,
    token: str | None,
    snapshot_download: Any | None = None,
    s3_client: Any | None = None,
) -> Mapping[str, object]:
    """Download one immutable snapshot and return only the staged manifest."""

    validate_immutable_revision(revision)
    if not isinstance(token, str) or not token.strip():
        raise CheckpointHandoffError("HF_TOKEN is required for checkpoint handoff")
    if snapshot_download is None:
        try:
            from huggingface_hub import (  # type: ignore[import-not-found]
                snapshot_download as hub_snapshot_download,
            )
        except ImportError as exc:  # pragma: no cover - image dependency contract
            raise CheckpointHandoffError("huggingface_hub is required") from exc
        snapshot_download = hub_snapshot_download
    try:
        with tempfile.TemporaryDirectory(prefix="functiongemma-handoff-") as directory:
            snapshot_download(
                repo_id=TARGET_MODEL_ID,
                revision=revision,
                local_dir=directory,
                token=token,
            )
            staged = stage_checkpoint(
                directory,
                bucket=bucket,
                revision=revision,
                prefix=prefix,
                region=region,
                s3_client=s3_client,
            )
    except (CheckpointStagingError, CheckpointHandoffError):
        raise
    except Exception as exc:
        raise CheckpointHandoffError(
            f"immutable FunctionGemma handoff failed: {type(exc).__name__}"
        ) from exc
    return staged.to_dict()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", default="post-training/checkpoints")
    parser.add_argument("--region", default=None)
    parser.add_argument("--token-env", default="HF_TOKEN")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = stage_from_huggingface(
            bucket=args.bucket,
            revision=args.revision,
            prefix=args.prefix,
            region=args.region,
            token=os.environ.get(args.token_env),
        )
    except (CheckpointHandoffError, CheckpointStagingError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
