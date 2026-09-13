from __future__ import annotations

from pathlib import Path


def test_trainer_image_smoke_is_cuda_qlora_one_step_and_offline() -> None:
    backend = Path(__file__).resolve().parents[1]
    dockerfile = (backend / "workers/trainer/Dockerfile").read_text()
    smoke = (backend / "workers/trainer/image_smoke.py").read_text()

    assert "COPY workers/trainer/image_smoke.py" in dockerfile
    assert "torch.cuda.is_available()" in smoke
    assert "import bitsandbytes" in smoke
    assert "load_in_4bit=True" in smoke
    assert "local_files_only=True" in smoke
    assert "loss.backward()" in smoke
    assert "optimizer.step()" in smoke
    assert "save_pretrained" in smoke
    assert "TRANSFORMERS_OFFLINE" in smoke
    assert "HF_HUB_OFFLINE" in smoke


def test_evaluator_image_smoke_loads_local_base_and_adapter_offline() -> None:
    backend = Path(__file__).resolve().parents[1]
    dockerfile = (backend / "workers/evaluator/Dockerfile").read_text()
    smoke = (backend / "workers/evaluator/image_smoke.py").read_text()

    assert "COPY workers/evaluator/image_smoke.py" in dockerfile
    assert "AutoProcessor.from_pretrained" in smoke
    assert "AutoModelForCausalLM.from_pretrained" in smoke
    assert "PeftModel.from_pretrained" in smoke
    assert smoke.count("local_files_only=True") >= 3
    assert "torch.inference_mode()" in smoke
    assert "TRANSFORMERS_OFFLINE" in smoke
    assert "HF_HUB_OFFLINE" in smoke


def test_aws_worker_image_instructions_build_and_smoke_linux_amd64() -> None:
    readme = (Path(__file__).resolve().parents[2] / "infra/cdk/README.md").read_text()

    assert readme.count("--platform linux/amd64") >= 3
    assert "smoke the trainer and evaluator images" in readme.lower()
    assert "--gpus all" in readme
    assert readme.count("--network none") >= 2
    assert "--entrypoint python" in readme
