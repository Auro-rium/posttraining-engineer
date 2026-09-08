"""
Main application entry point for the autonomous post-training engineer.
Updated for AWS Agents for Humans Hackathon with Strands Agents.
"""
import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from threading import Lock
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.api.continuous_post_training import install_post_training_api
from app.api.live_readiness import install_live_readiness_api
from app.api.run_comparison import install_run_comparison_api
from app.core.environment import create_service_recovery_environment
from app.core.orchestrator import create_orchestrator
from app.core.state import OptimizationRun
from app.observability import EventType, TelemetryRecorder
from app.posttraining.run_history import (
    MAX_RUNS,
    RunHistoryRecord,
    RunHistoryRepository,
    RunLimitExceeded,
    RunRegistry,
)
from app.providers.repository import DynamoDBRunRepository
from app.providers.sagemaker import SageMakerProvider
from app.runtime_config import get_runtime_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _LocalRunHistoryRepository:
    """Process-local history adapter for the no-AWS demo mode.

    The API uses the same :class:`RunRegistry` contract in local and AWS
    modes.  This adapter deliberately does not pretend to be durable: it is
    only a development/demo store and is replaced by DynamoDB in AWS mode.
    """

    def __init__(self) -> None:
        self._records: dict[str, RunHistoryRecord] = {}
        self._lock = Lock()

    def reserve_run(
        self, record: RunHistoryRecord, *, max_runs: int = MAX_RUNS
    ) -> RunHistoryRecord:
        with self._lock:
            if record.run_id in self._records:
                raise ValueError(f"run already exists: {record.run_id}")
            if len(self._records) >= max_runs:
                raise RunLimitExceeded(f"maximum of {max_runs} runs reached")
            expected_number = len(self._records) + 1
            if record.run_number != expected_number:
                raise ValueError(
                    f"expected next run number {expected_number}, got {record.run_number}"
                )
            self._records[record.run_id] = record
        return record

    def get_run(self, run_id: str) -> RunHistoryRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def get_history_run(self, run_id: str) -> RunHistoryRecord | None:
        return self.get_run(run_id)

    def list_runs(self, *, limit: int = MAX_RUNS) -> Sequence[RunHistoryRecord]:
        if not 1 <= limit <= MAX_RUNS:
            raise ValueError(f"limit must be between 1 and {MAX_RUNS}")
        with self._lock:
            records = sorted(self._records.values(), key=lambda item: item.run_number)
            return tuple(records[-limit:])

    def list_history_runs(self, *, limit: int = MAX_RUNS) -> Sequence[RunHistoryRecord]:
        return self.list_runs(limit=limit)


def _create_run_registry(config: Any) -> RunRegistry:
    """Build the durable-history boundary without creating cloud resources."""

    repository: RunHistoryRepository
    if config.app_mode == "aws":
        # Constructing this adapter is side-effect free.  The DynamoDB table
        # must already exist and is supplied by deployment/CDK configuration.
        repository = DynamoDBRunRepository(
            table_name=config.dynamodb_table_name,
            region_name=config.aws_region,
        )
    else:
        repository = _LocalRunHistoryRepository()
    return RunRegistry(repository, max_runs=MAX_RUNS)

# Validate deployment configuration before creating the application. AWS mode
# fails closed rather than silently running the local simulation.
settings = get_runtime_config()

def _create_application_orchestrator(config: Any) -> Any:
    """Wire configured model/provider dependencies into the coordinator.

    SageMaker's client is lazy, so constructing this adapter does not create a
    job or another AWS resource.  The objective-worker adapter is intentionally
    left absent until one is explicitly configured; benchmark execution then
    fails closed rather than falling back to explanatory metrics.
    """

    provider = (
        SageMakerProvider(region_name=config.aws_region)
        if config.app_mode == "aws"
        else None
    )
    return create_orchestrator(
        model=config.strands_model,
        training_adapter=provider,
        evaluation_adapter=provider,
    )


# Initialize the orchestrator with the same model and provider configuration
# advertised by the process health contract.
orchestrator = _create_application_orchestrator(settings)

# In-memory store for active runs (would use database in production)
active_runs: dict[str, OptimizationRun] = {}

# Create FastAPI app
app = FastAPI(
    title="Autonomous Post-Training Engineer for AWS Agents for Humans",
    description="End-to-end autonomous post-training optimization using Strands Agents",
    version="0.1.0",
)
install_post_training_api(app)
install_live_readiness_api(app)
app.state.run_registry = _create_run_registry(settings)
app.state.telemetry = TelemetryRecorder()
app.state.run_numbers = {}
install_run_comparison_api(app, app.state.run_registry)


def _record_telemetry(
    event_type: EventType,
    run: OptimizationRun,
    *,
    phase: str | None = None,
    job_id: str | None = None,
    evidence_label: str | None = None,
    status: str | None = None,
    attributes: Mapping[str, object] | None = None,
) -> None:
    """Emit correlation metadata while keeping telemetry non-blocking."""

    try:
        app.state.telemetry.record(
            event_type,
            run_id=run.runId,
            run_number=app.state.run_numbers.get(run.runId, 1),
            experiment_id=f"{run.runId}-experiment",
            phase=phase,
            job_id=job_id,
            evidence_label=evidence_label,
            status=status,
            attributes=attributes,
        )
    except Exception:
        # The recorder itself isolates sink failures; this boundary also
        # protects legacy run objects and route execution from bad metadata.
        logger.exception("Failed to emit post-training telemetry")


def _record_phase_telemetry(
    run: OptimizationRun,
    phase: str,
    phase_result: Mapping[str, Any] | None,
    *,
    started: bool = False,
) -> None:
    """Record phase/job lifecycle metadata without recording model content."""

    if started:
        _record_telemetry(EventType.PHASE_STARTED, run, phase=phase, status="running")
        if phase in {"execute_training", "evaluate"}:
            _record_telemetry(EventType.JOB_SUBMITTED, run, phase=phase, status="submitted")
        return

    result = phase_result or {}
    result_status = str(result.get("status", "unknown"))
    event_type = (
        EventType.PHASE_COMPLETED
        if result_status == "completed"
        else EventType.PHASE_FAILED
    )
    _record_telemetry(event_type, run, phase=phase, status=result_status)
    if phase in {"execute_training", "evaluate"}:
        job_event = (
            EventType.JOB_COMPLETED
            if result_status == "completed"
            else EventType.JOB_FAILED
        )
        _record_telemetry(job_event, run, phase=phase, status=result_status)
    if phase == "promote_decision" and result_status == "completed":
        output = str(result.get("output", ""))
        decision = "PROMOTE" if "PROMOTE" in output else "REJECT"
        _record_telemetry(
            EventType.PROMOTION_DECIDED,
            run,
            phase=phase,
            evidence_label="EXPLANATION",
            status=decision,
            attributes={"decision": decision},
        )



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "autonomous-post-training-engineer",
            "version": "0.1.0",
            "mode": settings.app_mode,
            "role": settings.service_role,
            "aws_region": settings.aws_region,
            "reasoning_model": settings.strands_model,
            "target_model": settings.target_model,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "orchestrator": "ready",
                "strands_agents": "initialized",
                "sagemaker_provider": (
                    "configured" if settings.app_mode == "aws" else "not_configured"
                ),
                "objective_worker": "not_configured",
                "environment": "available",
                "run_history": "ready",
                "telemetry": "ready",
            }
        }
    )


@app.get("/.well-known/agent-card.json")
async def agent_card():
    """Publish a minimal role card for internal service discovery."""
    return {
        "name": f"autonomous-post-training-{settings.service_role}",
        "description": "Strands post-training specialist service",
        "url": "/",
        "capabilities": ["typed-post-training-operations"],
        "role": settings.service_role,
    }


@app.post("/api/runs")
async def create_optimization_run(
    target_model: str,
    base_checkpoint: str,
    environment: str = "agentgym-service-recovery",
    objective: str = "maximize task success rate",
    budget: dict | None = None
):
    """
    Create a new optimization run.

    Args:
        target_model: Gemma model to optimize (e.g., "google/gemma-2-9b-it")
        base_checkpoint: Starting model checkpoint (S3 URI or Hugging Face repo)
        environment: AgentGym environment to use
        objective: Optimization objective
        budget: Resource constraints
    """
    try:
        # Apply the deployment's hard budget defaults and reject request-level
        # values that would widen them.  This keeps the HTTP boundary aligned
        # with RuntimeConfig rather than allowing a caller to bypass it.
        if budget is None:
            budget = {
                "maxExperiments": settings.max_experiments,
                "maxCostUSD": settings.max_cost_usd,
                "maxTrainingTimeMin": settings.max_training_time_min,
            }
        else:
            budget = dict(budget)
        max_experiments = budget.get("maxExperiments", settings.max_experiments)
        max_cost_usd = budget.get("maxCostUSD", settings.max_cost_usd)
        max_training_time = budget.get(
            "maxTrainingTimeMin", settings.max_training_time_min
        )
        if (
            isinstance(max_experiments, bool)
            or not isinstance(max_experiments, int)
            or not 1 <= max_experiments <= settings.max_experiments
        ):
            raise ValueError(
                f"maxExperiments must be an integer between 1 and {settings.max_experiments}"
            )
        if (
            isinstance(max_cost_usd, bool)
            or not isinstance(max_cost_usd, (int, float))
            or not math.isfinite(float(max_cost_usd))
            or not 0 <= float(max_cost_usd) <= settings.max_cost_usd
        ):
            raise ValueError(
                f"maxCostUSD must be finite and between 0 and {settings.max_cost_usd}"
            )
        if (
            isinstance(max_training_time, bool)
            or not isinstance(max_training_time, int)
            or not 1 <= max_training_time <= settings.max_training_time_min
        ):
            raise ValueError(
                "maxTrainingTimeMin must be a positive integer within the configured limit"
            )
        budget.update(
            {
                "maxExperiments": max_experiments,
                "maxCostUSD": float(max_cost_usd),
                "maxTrainingTimeMin": max_training_time,
            }
        )

        # Generate a unique run ID and reserve the bounded comparison slot.
        run_id = (
            f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid4().hex[:8]}"
        )
        history_records = app.state.run_registry.list_runs()
        run_number = len(history_records) + 1
        parent_run_id = history_records[-1].run_id if history_records else None

        # Initialize the run
        run = orchestrator.initialize_run(
            run_id=run_id,
            target_model=target_model,
            base_checkpoint=base_checkpoint,
            environment=environment,
            objective=objective,
            budget=budget
        )

        app.state.run_registry.register(
            RunHistoryRecord(
                run_id=run_id,
                run_number=run_number,
                parent_run_id=parent_run_id,
                champion_run_id=parent_run_id,
                model_id=target_model,
            )
        )
        app.state.run_numbers[run_id] = run_number

        # Store the run
        active_runs[run_id] = run

        _record_telemetry(
            EventType.RUN_STARTED,
            run,
            status=run.status,
            attributes={"environment": environment, "target_model": target_model},
        )

        logger.info(f"Created optimization run {run_id}")

        return JSONResponse(
            status_code=201,
            content={
                "runId": run_id,
                "status": "created",
                "targetModel": target_model,
                "environment": environment,
                "createdAt": run.createdAt.isoformat(),
                "message": "Optimization run created successfully. Use POST /api/runs/{run_id}/step to begin execution."
            }
        )

    except RunLimitExceeded as e:
        logger.warning("Run history limit reached: %s", e)
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to create optimization run: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to create run: {str(e)}")


@app.post("/api/runs/{run_id}/step")
async def execute_next_step(run_id: str):
    """
    Execute the next step in the optimization workflow.

    Args:
        run_id: The optimization run identifier
    """
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = active_runs[run_id]

    try:
        # Execute exactly one next phase; /auto is reserved for the full loop.
        current_index = orchestrator.phases.index(run.currentPhase) if run.currentPhase in orchestrator.phases else -1
        if current_index + 1 >= len(orchestrator.phases):
            return JSONResponse(content={"runId": run_id, "status": run.status, "message": "run is complete"})
        next_phase = orchestrator.phases[current_index + 1]
        _record_phase_telemetry(run, next_phase, None, started=True)
        updated_run, workflow_result = orchestrator.execute_workflow(
            run, target_phase=next_phase
        )

        # Update stored run
        active_runs[run_id] = updated_run
        phase_results = workflow_result.get("phase_results", [])
        phase_result = phase_results[0] if phase_results else None
        _record_phase_telemetry(updated_run, next_phase, phase_result)

        logger.info(f"Executed step for run {run_id}: {workflow_result.get('overall_status')}")

        return JSONResponse(
            content={
                "runId": run_id,
                "currentPhase": updated_run.currentPhase,
                "status": updated_run.status,
                "workflowResult": workflow_result,
                "updatedAt": updated_run.updatedAt.isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to execute step for run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute step: {str(e)}")


@app.post("/api/runs/{run_id}/auto")
async def execute_auto_workflow(run_id: str, background_tasks: BackgroundTasks):
    """
    Execute the full optimization workflow automatically.

    Args:
        run_id: The optimization run identifier
    """
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = active_runs[run_id]

    # Add the full workflow execution as a background task
    def run_full_workflow():
        try:
            current_index = (
                orchestrator.phases.index(run.currentPhase)
                if run.currentPhase in orchestrator.phases
                else -1
            )
            for phase in orchestrator.phases[current_index + 1 :]:
                _record_phase_telemetry(run, phase, None, started=True)
            final_run, workflow_result = orchestrator.execute_workflow(run)
            active_runs[run_id] = final_run
            for phase_result in workflow_result.get("phase_results", []):
                if isinstance(phase_result, Mapping):
                    _record_phase_telemetry(
                        final_run,
                        str(phase_result.get("phase", final_run.currentPhase)),
                        phase_result,
                    )
            terminal_event = (
                EventType.RUN_COMPLETED
                if workflow_result.get("overall_status") == "completed"
                else EventType.RUN_FAILED
            )
            _record_telemetry(terminal_event, final_run, status=final_run.status)
            logger.info(f"Completed auto workflow for run {run_id}: {workflow_result.get('overall_status')}")
        except Exception as e:
            _record_telemetry(EventType.RUN_FAILED, run, phase=run.currentPhase, status="failed")
            logger.error(f"Auto workflow failed for run {run_id}: {str(e)}")

    background_tasks.add_task(run_full_workflow)

    return JSONResponse(
        content={
            "runId": run_id,
            "status": "workflow_started",
            "message": "Full optimization workflow started in background",
            "startedAt": datetime.utcnow().isoformat()
        }
    )


@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str):
    """
    Get the current status of an optimization run.

    Args:
        run_id: The optimization run identifier
    """
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = active_runs[run_id]

    return JSONResponse(
        content={
            "runId": run.runId,
            "targetModel": run.targetModel,
            "environment": run.environment,
            "objective": run.objective,
            "currentPhase": run.currentPhase,
            "status": run.status,
            "createdAt": run.createdAt.isoformat(),
            "updatedAt": run.updatedAt.isoformat(),
            "baselinePerformance": run.baselinePerformance,
            "championPerformance": run.championPerformance,
            "totalExperiments": len(run.experiments),
            "totalCandidates": len(run.candidates),
            "championCheckpoint": run.championCheckpoint
        }
    )


@app.get("/api/runs/{run_id}/experiments")
async def get_run_experiments(run_id: str):
    """
    Get experiment history for an optimization run.

    Args:
        run_id: The optimization run identifier
    """
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = active_runs[run_id]

    return JSONResponse(
        content={
            "runId": run_id,
            "experiments": run.experiments,
            "candidates": run.candidates,
            "datasets": run.datasets,
            "failureClusters": run.failureClusters,
            "hypotheses": run.hypotheses,
            "trajectories": run.trajectories,
            "championCheckpoint": run.championCheckpoint,
            "championPerformance": run.championPerformance,
            "baselinePerformance": run.baselinePerformance
        }
    )


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """
    Cancel an optimization run.

    Args:
        run_id: The optimization run identifier
    """
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = active_runs[run_id]
    run.status = "cancelled"
    run.update_timestamp()
    _record_telemetry(EventType.RUN_FAILED, run, status="cancelled")

    logger.info(f"Cancelled run {run_id}")

    return JSONResponse(
        content={
            "runId": run_id,
            "status": "cancelled",
            "message": "Optimization run cancelled",
            "cancelledAt": run.updatedAt.isoformat()
        }
    )


@app.post("/api/demo/reset-environment")
async def reset_environment():
    """
    Reset the service recovery environment for demonstration purposes.
    """
    try:
        env = create_service_recovery_environment("demo-env-001")
        observation = env.reset()

        return JSONResponse(
            content={
                "message": "Environment reset successfully",
                "environmentId": "demo-env-001",
                "initialObservation": observation,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Failed to reset environment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
