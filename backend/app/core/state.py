"""
OptimizationRun state model for the autonomous post-training engineer.
Defines the central state that all agents read from and write to.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class OptimizationRun(BaseModel):
    """Central state object that flows between all agents."""

    # Identifiers
    runId: str = Field(..., description="Unique identifier for this optimization run")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    # Configuration
    targetModel: str = Field(..., description="Gemma model to optimize (e.g., 'google/gemma-2-9b-it')")
    baseCheckpoint: str = Field(..., description="S3 URI or Hugging Face repo of base model")
    environment: str = Field(..., description="AgentGym environment (e.g., 'agentgym-service-recovery')")
    objective: str = Field(..., description="Optimization objective (e.g., 'maximize task success rate')")
    budget: Dict[str, Any] = Field(..., description="Resource constraints")

    # Current State
    championCheckpoint: Optional[str] = Field(None, description="S3 URI of best performing model")
    currentPhase: str = Field("initialized", description="Current phase of optimization")
    status: str = Field("running", description="Overall status: running, completed, failed, cancelled")

    # Experimental Artifacts (stored as references to S3)
    trajectories: List[str] = Field(default_factory=list, description="S3 URIs of trajectory data")
    failureClusters: List[str] = Field(default_factory=list, description="S3 URIs of failure analysis")
    hypotheses: List[str] = Field(default_factory=list, description="S3 URIs of research hypotheses")
    datasets: List[str] = Field(default_factory=list, description="S3 URIs of verified SFT datasets")
    experiments: List[str] = Field(default_factory=list, description="S3 URIs of training experiments")
    candidates: List[str] = Field(default_factory=list, description="S3 URIs of model checkpoints")

    # Constraints & Gates
    limits: Dict[str, Any] = Field(default_factory=dict, description="Derived from budget")
    gates: Dict[str, float] = Field(
        default_factory=lambda: {"minImprovement": 0.10, "maxRegression": 0.05},
        description="Promotion gates: minimum improvement, maximum regression allowed"
    )

    # Metrics
    baselinePerformance: float = Field(0.0, description="Performance of base model")
    candidatePerformance: Optional[float] = Field(
        None, description="Measured performance of the latest evaluated candidate"
    )
    championPerformance: float = Field(0.0, description="Performance of current champion")
    totalCostUSD: float = Field(0.0, description="Total cost in USD")
    totalTrainingTimeMin: int = Field(0, description="Total training time in minutes")

    def update_timestamp(self):
        """Update the updatedAt timestamp."""
        self.updatedAt = datetime.utcnow()
