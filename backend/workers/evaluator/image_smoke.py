"""Load a local FunctionGemma base plus adapter and run one offline forward pass.

This smoke uses no sealed tasks and creates no evaluation report. It validates
only that the evaluator image can load a staged base and local PEFT adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _tool_schemas() -> list[dict[str, Any]]:
    from app.objective.models import ALLOWED_TOOLS

    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Service-recovery tool: {name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
        }
        for name in ALLOWED_TOOLS
    ]


def _directory(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{name} must be an existing local directory")
    return path.resolve()


def run_smoke(base_model_dir: str, adapter_dir: str, device: str = "cpu") -> dict[str, object]:
    """Load a local base and PEFT adapter, then execute one safe model forward."""

    try:
        import torch  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError("evaluator image is missing Transformers/PEFT dependencies") from exc

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluator smoke requested CUDA but no GPU is available")
    base_path = _directory(base_model_dir, "base_model_dir")
    adapter_path = _directory(adapter_dir, "adapter_dir")
    target = torch.device(device)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        local_files_only=True,
        is_trainable=False,
    ).to(target)
    model.eval()

    encoded = processor.apply_chat_template(
        [
            {"role": "developer", "content": "Use the provided service-recovery functions."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_id": "evaluator-image-smoke",
                        "objective": "inspect the service logs",
                        "service": "api",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        tools=_tool_schemas(),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(target) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    with torch.inference_mode():
        logits = model(**inputs).logits
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("evaluator smoke produced non-finite model output")
    return {"status": "PASS", "device": str(target), "forward_passes": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(args.base_model_dir, args.adapter_dir, args.device), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
