"""Live, non-destructive test of all eight Strands specialists via Bedrock."""

from __future__ import annotations

import hashlib
import io
import os
from contextlib import redirect_stdout
from dataclasses import dataclass

from app.agents.prompt_contract import NEMOTRON_MODEL_ID, resolve_nemotron_model
from app.providers.bedrock import BedrockStrandsModel


@dataclass(frozen=True)
class Specialist:
    name: str
    prompt: str


SPECIALISTS = (
    Specialist(
        "BenchmarkAgent",
        (
            "Return JSON with keys role, task, and status. State that you would measure "
            "a service-recovery baseline without inventing metrics."
        ),
    ),
    Specialist(
        "FailureAnalystAgent",
        (
            "Return JSON with keys role, task, and status. Name one observable failure "
            "category and say that it requires trajectory evidence."
        ),
    ),
    Specialist(
        "ResearchAgent",
        (
            "Return JSON with keys role, hypothesis, and falsifiable. Give one falsifiable "
            "hypothesis about verification-after-repair behavior."
        ),
    ),
    Specialist(
        "DataCuratorAgent",
        (
            "Return JSON with keys role, task, and verification_required. State that proposed "
            "repair rows require environment replay before training admission."
        ),
    ),
    Specialist(
        "TrainingDesignerAgent",
        (
            "Return JSON with keys role, rank, learning_rate, epochs, and bounded. Choose "
            "only rank 8/16/32/64, learning rate 0.0001/0.0002/0.0005/0.001, and epochs 1-4."
        ),
    ),
    Specialist(
        "TrainingExecutorAgent",
        (
            "Return JSON with keys role, task, and provider_backed. State that training "
            "artifacts require a real provider job ID."
        ),
    ),
    Specialist(
        "EvalAgent",
        (
            "Return JSON with keys role, task, and independent. State that candidate evaluation "
            "must use identical sealed inputs and must not decide promotion."
        ),
    ),
    Specialist(
        "ChampionManagerAgent",
        (
            "Return JSON with keys role, decision, and deterministic. State that promotion "
            "requires deterministic improvement and regression gates."
        ),
    ),
)


def main() -> int:
    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = resolve_nemotron_model(os.getenv("STRANDS_MODEL", NEMOTRON_MODEL_ID))
    model_provider = BedrockStrandsModel(model_id, region_name=region)
    results: list[dict[str, str]] = []

    for specialist in SPECIALISTS:
        agent = model_provider.create_agent(
            name=specialist.name,
            system_prompt=(
                f"You are {specialist.name}, a bounded professional post-training specialist. "
                "Answer only the requested task. Do not claim live metrics, artifacts, training, "
                "or evaluation unless they are actually supplied."
            ),
        )
        # Strands may echo a response through its default callback; capture
        # that stream so this script remains metadata-only as well.
        with redirect_stdout(io.StringIO()):
            response = str(agent(specialist.prompt)).strip()
        if not response:
            raise RuntimeError(f"{specialist.name} returned an empty response")
        # Keep the live script metadata-only: never print prompts or model
        # completions into a terminal/log that may be retained as evidence.
        results.append(
            {
                "agent": specialist.name,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "response_chars": str(len(response)),
            }
        )

    print({
        "status": "passed",
        "model_id": model_id,
        "region": region,
        "agents_tested": len(results),
        "resources_created": [],
        "results": results,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
