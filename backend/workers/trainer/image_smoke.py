"""Run one offline QLoRA optimizer step against the staged FunctionGemma base.

This image-validation entrypoint is not the SageMaker training entrypoint and
does not produce a promotion-eligible artifact. It exists to catch CUDA,
bitsandbytes, local-checkpoint, and PEFT integration failures before training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Make accidental Hub fallback impossible even if a future loader omits its
# explicit local_files_only argument.
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


def run_smoke(base_model_dir: str, output_dir: str) -> dict[str, object]:
    """Load the staged base in 4-bit and perform exactly one LoRA update."""

    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]  # noqa: F401
        import torch  # type: ignore[import-not-found]
        from peft import (  # type: ignore[import-not-found]
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("trainer image is missing CUDA QLoRA dependencies") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("trainer image smoke requires an available CUDA GPU")
    base = _directory(base_model_dir, "base_model_dir")
    output = Path(output_dir).expanduser().resolve()
    if output == base or base in output.parents:
        raise ValueError("smoke output must not be inside or overwrite the base checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("smoke output directory must be empty")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    processor = AutoProcessor.from_pretrained(
        base, local_files_only=True, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    encoded = processor.apply_chat_template(
        [
            {"role": "developer", "content": "Use the provided service-recovery functions."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_id": "trainer-image-smoke",
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
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("trainer smoke produced no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output_values = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    loss = output_values.loss
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError("trainer smoke produced a non-finite loss")
    loss.backward()
    if not any(parameter.grad is not None for parameter in trainable):
        raise RuntimeError("trainer smoke produced no LoRA gradients")
    optimizer.step()
    model.save_pretrained(output)
    if not (output / "adapter_config.json").is_file() or not any(
        path.is_file() and path.stat().st_size > 0
        for suffix in ("*.safetensors", "*.bin")
        for path in output.glob(suffix)
    ):
        raise RuntimeError("trainer smoke did not write a loadable adapter")
    return {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "optimizer_steps": 1,
        "adapter_written": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.base_model_dir, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
