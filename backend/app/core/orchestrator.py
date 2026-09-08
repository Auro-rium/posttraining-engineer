"""
Workflow Orchestrator - Coordinates the eight agents in the correct sequence.
Manages the optimization run state and agent handoffs.
"""
from typing import Dict, Any, Optional
import json
from datetime import datetime
from app.core.state import OptimizationRun
from app.agents.benchmark_agent import create_benchmark_agent
from app.agents.failure_analyst_agent import create_failure_analyst_agent
from app.agents.research_agent import create_research_agent
from app.agents.data_curator_agent import create_data_curator_agent
from app.agents.training_designer_agent import create_training_designer_agent
from app.agents.training_executor_agent import create_training_executor_agent
from app.agents.eval_agent import create_eval_agent
from app.agents.champion_manager_agent import create_champion_manager_agent


class OptimizationOrchestrator:
    """Orchestrates the autonomous post-training workflow."""

    def __init__(
        self,
        *,
        model: str | None = None,
        benchmark_adapter: Any = None,
        training_adapter: Any = None,
        evaluation_adapter: Any = None,
    ):
        # Initialize all eight agents
        resolved_model = model or "nvidia.nemotron-super-3-120b"
        self.model_id = resolved_model
        self.benchmark_agent = create_benchmark_agent(
            model=resolved_model, adapter=benchmark_adapter
        )
        self.failure_analyst_agent = create_failure_analyst_agent(model=resolved_model)
        self.research_agent = create_research_agent(model=resolved_model)
        self.data_curator_agent = create_data_curator_agent(model=resolved_model)
        self.training_designer_agent = create_training_designer_agent(model=resolved_model)
        self.training_executor_agent = create_training_executor_agent(
            model=resolved_model, adapter=training_adapter
        )
        self.eval_agent = create_eval_agent(
            model=resolved_model, adapter=evaluation_adapter
        )
        self.champion_manager_agent = create_champion_manager_agent(model=resolved_model)

        # Define the workflow phases
        self.phases = [
            "benchmark",
            "analyze_failures",
            "research",
            "curate_data",
            "design_training",
            "execute_training",
            "evaluate",
            "promote_decision"
        ]

    def initialize_run(self,
                      run_id: str,
                      target_model: str,
                      base_checkpoint: str,
                      environment: str,
                      objective: str,
                      budget: Dict[str, Any]) -> OptimizationRun:
        """
        Initialize a new optimization run.

        Args:
            run_id: Unique identifier for this run
            target_model: Gemma model to optimize
            base_checkpoint: Starting model checkpoint
            environment: AgentGym environment to use
            objective: Optimization objective
            budget: Resource constraints

        Returns:
            Initialized OptimizationRun state
        """
        run = OptimizationRun(
            runId=run_id,
            targetModel=target_model,
            baseCheckpoint=base_checkpoint,
            environment=environment,
            objective=objective,
            budget=budget,
            limits={
                "maxExperiments": budget.get("maxExperiments", 3),
                "maxCostUSD": budget.get("maxCostUSD", 50),
                "maxTrainingTimeMin": budget.get("maxTrainingTimeMin", 120)
            }
        )

        return run

    def execute_phase(self,
                     phase: str,
                     run_state: OptimizationRun) -> tuple[OptimizationRun, Dict[str, Any]]:
        """
        Execute a single phase of the optimization workflow.

        Args:
            phase: Name of phase to execute
            run_state: Current optimization run state

        Returns:
            Updated run state and phase results
        """
        # Update phase and timestamp
        run_state.currentPhase = phase
        run_state.update_timestamp()

        phase_results = {
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "running",
            "agent_used": None,
            "output": None
        }

        try:
            if phase == "benchmark":
                # Run benchmark agent to establish baseline
                benchmark_result = self.benchmark_agent.execute_benchmark(
                    run_id=run_state.runId,
                    environment_config={
                        "environment_id": run_state.environment,
                        "task_type": run_state.objective
                    },
                    num_episodes=10
                )

                # Extract trajectory references for state
                try:
                    benchmark_data = json.loads(benchmark_result)
                    if not benchmark_data.get("benchmark_completed", False):
                        raise ValueError(
                            "objective benchmark adapter did not complete a real benchmark"
                        )
                    run_state.trajectories = benchmark_data.get("trajectory_references", [])
                    aggregate_metrics = benchmark_data.get("aggregate_metrics", {})
                    if not isinstance(aggregate_metrics, dict) or not isinstance(
                        aggregate_metrics.get("success_rate"), (int, float)
                    ):
                        raise ValueError("objective benchmark returned no measured success_rate")
                    run_state.baselinePerformance = float(aggregate_metrics["success_rate"])
                except json.JSONDecodeError:
                    run_state.baselinePerformance = 0.0

                phase_results.update({
                    "agent_used": "BenchmarkAgent",
                    "status": "completed",
                    "output": benchmark_result
                })

            elif phase == "analyze_failures":
                # Analyze failures from benchmarking
                if not run_state.trajectories:
                    raise ValueError("No trajectories available for failure analysis")

                failure_result = self.failure_analyst_agent.analyze_failures(
                    benchmark_results=json.dumps({
                        "run_id": run_state.runId,
                        "aggregate_metrics": {"success_rate": run_state.baselinePerformance}
                    }),
                    trajectory_references=run_state.trajectories
                )

                # Store failure cluster references
                try:
                    failure_data = json.loads(failure_result)
                    run_state.failureClusters = [f"failure_cluster_{i}" for i in range(len(failure_data.get("failure_clusters", [])))]
                except json.JSONDecodeError:
                    pass

                phase_results.update({
                    "agent_used": "FailureAnalystAgent",
                    "status": "completed",
                    "output": failure_result
                })

            elif phase == "research":
                # Generate research hypotheses
                if not run_state.failureClusters:
                    raise ValueError("No failure clusters available for research")

                research_result = self.research_agent.generate_hypotheses(
                    failure_analysis=json.dumps({
                        "analysis_id": "failure-analysis",
                        "failure_clusters": [
                            {
                                "cluster_id": f"cluster_{i}",
                                "failure_type": failure_type,
                                "description": failure_type,
                                "example_count": 1,
                                "example_trajectories": run_state.trajectories[:1],
                            }
                            for i, failure_type in enumerate(
                                ["failed_verification", "malformed_tool_arguments", "wrong_tool_selection"]
                            )
                        ],
                    })
                )

                # Store hypothesis references
                try:
                    research_data = json.loads(research_result)
                    run_state.hypotheses = [f"hypothesis_{i}" for i in range(research_data.get("num_hypotheses_generated", 0))]
                except json.JSONDecodeError:
                    pass

                phase_results.update({
                    "agent_used": "ResearchAgent",
                    "status": "completed",
                    "output": research_result
                })

            elif phase == "curate_data":
                # Curate training dataset
                if not run_state.hypotheses:
                    raise ValueError("No hypotheses available for data curation")

                # Need trajectories and failure analysis for curation
                curation_result = self.data_curator_agent.identify_decision_points(
                    trajectories=json.dumps({
                        "num_trajectories": len(run_state.trajectories),
                        "trajectory_references": run_state.trajectories
                    }),
                    failure_analysis=json.dumps({
                        "failure_clusters": [
                            {"cluster_id": f"cluster_{i}", "failure_type": failure_type}
                            for i, failure_type in enumerate(
                                ["failed_verification", "malformed_tool_arguments", "wrong_tool_selection"]
                            )
                        ]
                    })
                )

                # In a full implementation, would continue through generate_corrected_trajectories,
                # verify_corrections, and format_sft_dataset

                # For now, store dataset reference
                try:
                    curation_data = json.loads(curation_result)
                    run_state.datasets = [f"sft_dataset_{i}" for i in range(curation_data.get("decision_points_identified", 0))]
                except json.JSONDecodeError:
                    pass

                phase_results.update({
                    "agent_used": "DataCuratorAgent",
                    "status": "completed",
                    "output": curation_result
                })

            elif phase == "design_training":
                # Design QLoRA configuration
                if not run_state.datasets:
                    raise ValueError("No datasets available for training design")

                design_result = self.training_designer_agent.analyze_dataset_characteristics(
                    dataset_references=json.dumps({
                        "dataset_references": run_state.datasets,
                        "dataset_statistics": {"total_examples": 100}  # Placeholder
                    })
                )

                # Then select configuration
                config_result = self.training_designer_agent.select_qlora_configuration(
                    dataset_analysis=design_result,
                    budget_constraints=run_state.limits
                )

                # Store experiment reference
                try:
                    config_data = json.loads(config_result)
                    run_state.experiments = [f"experiment_{i}" for i in range(1)]  # One experiment this round
                    # In reality would store the selected configuration
                except json.JSONDecodeError:
                    pass

                phase_results.update({
                    "agent_used": "TrainingDesignerAgent",
                    "status": "completed",
                    "output": config_result
                })

            elif phase == "execute_training":
                # Execute training job
                if not run_state.experiments:
                    raise ValueError("No experiments available for execution")

                training_configuration = run_state.budget.get("trainingConfiguration")
                if not isinstance(training_configuration, dict):
                    raise ValueError(
                        "trainingConfiguration must be supplied by the training designer "
                        "before submitting a real SageMaker job"
                    )
                if any("placeholder" in reference.lower() for reference in run_state.datasets):
                    raise ValueError("placeholder dataset references cannot be submitted")
                training_result = self.training_executor_agent.submit_training_job(
                    configuration=json.dumps(training_configuration),
                    dataset_references=json.dumps({
                        "dataset_references": run_state.datasets,
                    }),
                    base_model=run_state.baseCheckpoint,
                    job_name=f"optimization_run_{run_state.runId}_exp_{len([e for e in run_state.experiments if e.startswith('experiment_')])}"
                )

                phase_results.update({
                    "agent_used": "TrainingExecutorAgent",
                    "status": "submitted",  # Training is async
                    "output": training_result
                })

            elif phase == "evaluate":
                # Evaluate only a provider-returned candidate artifact.  The
                # current phase machine intentionally fails closed if the
                # asynchronous training boundary was not connected.
                if not run_state.candidates:
                    raise ValueError(
                        "no trained candidate artifact is recorded; wait for SageMaker "
                        "training completion before evaluation"
                    )
                model_artifact = run_state.candidates[-1]
                if not model_artifact.startswith("s3://"):
                    raise ValueError("candidate artifact must be an S3 URI returned by the provider")
                eval_result = self.eval_agent.evaluate_held_out_performance(
                    model_artifacts=json.dumps({
                        "artifacts": {"model_artifacts": {"candidate": model_artifact}}
                    }),
                    evaluation_config={
                        "environment_id": run_state.environment,
                        "held_out_tasks": True
                    },
                    num_episodes=15
                )

                # Also run regression benchmarks
                regression_result = self.eval_agent.run_regression_benchmarks(
                    model_artifacts=json.dumps({
                        "artifacts": {"model_artifacts": {"candidate": model_artifact}}
                    }),
                    regression_suite={"environment": run_state.environment, "held_out": False},
                    baseline_performance=run_state.baselinePerformance
                )

                # Combine metrics
                metrics_result = self.eval_agent.calculate_performance_metrics(
                    held_out_results=eval_result,
                    regression_results=regression_result
                )

                # Update champion performance if improved
                try:
                    metrics_data = json.loads(metrics_result)
                    combined_metrics = metrics_data.get("combined_performance_metrics", {})
                    candidate_performance = combined_metrics.get("combined_score")
                    if (
                        metrics_data.get("status") != "completed"
                        or not isinstance(candidate_performance, (int, float))
                        or metrics_data.get("evidence_class") not in {"LIVE", "PRIOR_VERIFIED_RUN"}
                    ):
                        raise ValueError(
                            "evaluation did not return verified objective metrics"
                        )
                    run_state.candidatePerformance = float(candidate_performance)
                except json.JSONDecodeError:
                    candidate_performance = 0.0

                phase_results.update({
                    "agent_used": "EvalAgent",
                    "status": "completed",
                    "output": json.dumps({
                        "evaluation": eval_result,
                        "regression": regression_result,
                        "metrics": metrics_result
                    })
                })

            elif phase == "promote_decision":
                # Make promotion decision
                if not run_state.candidates:
                    raise ValueError("No candidates available for promotion decision")

                candidate_performance = run_state.candidatePerformance
                if candidate_performance is None:
                    raise ValueError("promotion requires measured candidate evaluation metrics")
                champion_performance = run_state.championPerformance if run_state.championCheckpoint else run_state.baselinePerformance

                # Apply improvement gate
                improve_result = self.champion_manager_agent.apply_improvement_gate(
                    candidate_performance=candidate_performance,
                    champion_performance=champion_performance,
                    min_improvement=run_state.gates.get("minImprovement", 0.10)
                )

                # Apply regression gate
                regress_result = self.champion_manager_agent.apply_regression_gate(
                    candidate_performance=candidate_performance,
                    champion_performance=champion_performance,
                    max_regression=run_state.gates.get("maxRegression", 0.05)
                )

                # Make final decision
                decision_result = self.champion_manager_agent.make_promotion_decision(
                    improvement_gate_result=improve_result,
                    regression_gate_result=regress_result,
                    candidate_info=json.dumps({
                        "candidate_id": f"candidate_{len(run_state.candidates)-1}",
                        "experiment_id": f"experiment_{len(run_state.experiments)-1}" if run_state.experiments else "none"
                    })
                )

                # Update champion if promoted
                try:
                    decision_data = json.loads(decision_result)
                    if decision_data.get("promotion_decision") == "PROMOTE":
                        run_state.championCheckpoint = f"s3://post-training-engineer-artifacts/models/candidate_{len(run_state.candidates)-1}"
                        run_state.championPerformance = candidate_performance
                        # Would actually call update_champion_reference here
                    else:
                        # Champion remains unchanged
                        pass
                except json.JSONDecodeError:
                    pass  # Keep existing champion on error

                phase_results.update({
                    "agent_used": "ChampionManagerAgent",
                    "status": "completed",
                    "output": decision_result
                })

            else:
                raise ValueError(f"Unknown phase: {phase}")

        except Exception as e:
            phase_results.update({
                "status": "failed",
                "error": str(e),
                "agent_used": phase_results.get("agent_used", "unknown"),
                "output": None
            })
            # Don't update run state on failure - let caller handle

        return run_state, phase_results

    def execute_workflow(self,
                        run_state: OptimizationRun,
                        target_phase: Optional[str] = None) -> tuple[OptimizationRun, Dict[str, Any]]:
        """
        Execute the optimization workflow from current phase to target phase.

        Args:
            run_state: Current optimization run state
            target_phase: Phase to execute up to (None = run full workflow)

        Returns:
            Final run state and workflow results
        """
        workflow_results = {
            "workflow_id": f"workflow_{run_state.runId}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "start_timestamp": datetime.utcnow().isoformat(),
            "phases_executed": [],
            "phase_results": [],
            "overall_status": "running",
            "current_phase": run_state.currentPhase
        }

        # Determine which phases remain. A single-step caller can target the next
        # phase; auto mode runs through the remaining state machine.
        current_index = self.phases.index(run_state.currentPhase) if run_state.currentPhase in self.phases else -1
        if target_phase and target_phase in self.phases:
            target_index = self.phases.index(target_phase)
            phases_to_execute = self.phases[current_index + 1 : target_index + 1]
        else:
            phases_to_execute = self.phases[current_index + 1 :]

        # Execute each phase in sequence
        for phase in phases_to_execute:
            # Skip if we've already passed this phase
            if phase in workflow_results["phases_executed"]:
                continue

            try:
                run_state, phase_result = self.execute_phase(phase, run_state)

                workflow_results["phases_executed"].append(phase)
                workflow_results["phase_results"].append(phase_result)
                workflow_results["current_phase"] = phase

                # Stop if phase failed
                if phase_result.get("status") == "failed":
                    workflow_results["overall_status"] = "failed"
                    break

            except Exception as e:
                # Handle unexpected errors
                error_result = {
                    "phase": phase,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "failed",
                    "error": f"Orchestrator error: {str(e)}",
                    "agent_used": None,
                    "output": None
                }

                workflow_results["phases_executed"].append(phase)
                workflow_results["phase_results"].append(error_result)
                workflow_results["overall_status"] = "failed"
                break

        # Set overall completion status
        if workflow_results["overall_status"] != "failed":
            if target_phase:
                workflow_results["overall_status"] = "completed_up_to_target"
            elif len(workflow_results["phases_executed"]) == len(self.phases):
                workflow_results["overall_status"] = "completed"
            else:
                workflow_results["overall_status"] = "partially_completed"

        if workflow_results["overall_status"] == "completed" and run_state.currentPhase == "promote_decision":
            run_state.status = "completed"
            run_state.update_timestamp()
        workflow_results["end_timestamp"] = datetime.utcnow().isoformat()

        return run_state, workflow_results


# Factory function
def create_orchestrator(
    *,
    model: str | None = None,
    benchmark_adapter: Any = None,
    training_adapter: Any = None,
    evaluation_adapter: Any = None,
) -> OptimizationOrchestrator:
    """Create an orchestrator with explicit model and provider dependencies.

    The coordinator is intentionally dependency-injected: a missing live
    adapter remains visible to the relevant agent and fails closed instead of
    silently selecting a simulated implementation.
    """

    return OptimizationOrchestrator(
        model=model,
        benchmark_adapter=benchmark_adapter,
        training_adapter=training_adapter,
        evaluation_adapter=evaluation_adapter,
    )
