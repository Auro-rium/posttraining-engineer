"""
Main application entry point for the autonomous post-training engineer.
Updated for AWS Agents for Humans Hackathon with Strands Agents.
"""
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

# Import our new Strands-based components
from app.core.state import OptimizationRun
from app.core.orchestrator import create_orchestrator
from app.core.environment import create_service_recovery_environment
from app.runtime_config import get_runtime_config
from app.api.continuous_post_training import install_post_training_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate deployment configuration before creating the application. AWS mode
# fails closed rather than silently running the local simulation.
settings = get_runtime_config()

# Initialize the orchestrator
orchestrator = create_orchestrator()

# In-memory store for active runs (would use database in production)
active_runs: dict[str, OptimizationRun] = {}

# Create FastAPI app
app = FastAPI(
    title="Autonomous Post-Training Engineer for AWS Agents for Humans",
    description="End-to-end autonomous post-training optimization using Strands Agents",
    version="0.1.0",
)
install_post_training_api(app)



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
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "orchestrator": "ready",
                "strands_agents": "initialized",
                "environment": "available"
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
    budget: dict = None
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
        # Set default budget if not provided
        if budget is None:
            budget = {
                "maxExperiments": 3,
                "maxCostUSD": 50.0,
                "maxTrainingTimeMin": 120
            }

        # Generate unique run ID
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash((target_model, base_checkpoint)) % 10000:04d}"

        # Initialize the run
        run = orchestrator.initialize_run(
            run_id=run_id,
            target_model=target_model,
            base_checkpoint=base_checkpoint,
            environment=environment,
            objective=objective,
            budget=budget
        )

        # Store the run
        active_runs[run_id] = run

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
        updated_run, workflow_result = orchestrator.execute_workflow(
            run, target_phase=orchestrator.phases[current_index + 1]
        )

        # Update stored run
        active_runs[run_id] = updated_run

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
            final_run, workflow_result = orchestrator.execute_workflow(run)
            active_runs[run_id] = final_run
            logger.info(f"Completed auto workflow for run {run_id}: {workflow_result.get('overall_status')}")
        except Exception as e:
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
