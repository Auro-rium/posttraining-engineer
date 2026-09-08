"""
Research Agent - Generates testable hypotheses from failure analysis.
Responsible for proposing explanations that can be validated through experimentation.
"""
from strands import Agent, tool

from .prompt_contract import get_prompt_contract, resolve_agent_model
from strands.types.tools import ToolResult, ToolUse
from typing import Dict, Any, List
import json
from datetime import datetime
from app.core.state import OptimizationRun


class ResearchAgent:
    """Agent responsible for generating research hypotheses from failure analysis."""
    
    def __init__(self, model: str = None, model_provider: Any = None):
        prompt_contract = get_prompt_contract("ResearchAgent")
        self.agent = Agent(
            name="ResearchAgent",
            model=resolve_agent_model(model, model_provider=model_provider),
            system_prompt=prompt_contract.prompt,
        )
        
        # Register tools
        self.prompt_contract = prompt_contract
        self.prompt_metadata = prompt_contract.metadata()
        self.agent.tool_registry.register_tool(self.generate_hypotheses)
        self.agent.tool_registry.register_tool(self.validate_hypothesis_testability)
        self.agent.tool_registry.register_tool(self.design_validation_experiment)
    
    @tool
    def generate_hypotheses(self, 
                           failure_analysis: str,
                           max_hypotheses_per_cluster: int = 2) -> str:
        """
        Generate testable hypotheses from failure analysis.
        
        Args:
            failure_analysis: JSON string from Failure Analyst Agent
            max_hypotheses_per_cluster: Maximum hypotheses to generate per cluster
            
        Returns:
            JSON string containing generated hypotheses
        """
        # Parse failure analysis
        try:
            fail_data = json.loads(failure_analysis)
        except json.JSONDecodeError:
            fail_data = {"raw_data": failure_analysis}
        
        # In implementation, this would:
        # 1. Analyze each failure cluster
        # 2. Generate grounded, testable hypotheses
        # 3. Link each hypothesis to supporting evidence
        # 4. Ensure hypotheses are falsifiable
        
        # Generate sample hypotheses based on typical failure patterns
        hypotheses = []
        
        failure_clusters = fail_data.get("failure_clusters", [])
        for cluster in failure_clusters:
            failure_type = cluster.get("failure_type", "unknown")
            description = cluster.get("description", "")
            example_count = cluster.get("example_count", 0)
            
            # Generate hypotheses based on failure type
            if failure_type == "failed_verification":
                hypotheses.append({
                    "hypothesis_id": f"hyp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_001",
                    "cluster_id": cluster.get("cluster_id"),
                    "failure_type": failure_type,
                    "statement": "Gemma understands the repair sequence but frequently terminates before checking whether repairs were successful",
                    "evidence_references": cluster.get("example_trajectories", [])[:2],
                    "testable_prediction": "If Gemma is trained to always verify after repairs, success rate should increase significantly",
                    "falsifiable_criterion": "Verification behavior does not improve after training on verified sequences",
                    "confidence": 0.85
                })
                
                if len(hypotheses) < max_hypotheses_per_cluster * len(failure_clusters):
                    hypotheses.append({
                        "hypothesis_id": f"hyp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_002",
                        "cluster_id": cluster.get("cluster_id"),
                        "failure_type": failure_type,
                        "statement": "Gemma lacks understanding that restarting a service requires verification of the underlying fix",
                        "evidence_references": cluster.get("example_trajectories", [])[:1],
                        "testable_prediction": "Training on complete repair-verification sequences will reduce premature completion",
                        "falsifiable_criterion": "Premature completion rate remains unchanged after training",
                        "confidence": 0.75
                    })
            
            elif failure_type == "malformed_tool_arguments":
                hypotheses.append({
                    "hypothesis_id": f"hyp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_003",
                    "cluster_id": cluster.get("cluster_id"),
                    "failure_type": failure_type,
                    "statement": "Gemma has difficulty generating correctly formatted JSON for tool arguments, particularly with nested structures and escaping",
                    "evidence_references": cluster.get("example_trajectories", [])[:2],
                    "testable_prediction": "Providing Gemma with JSON formatting examples in training data will improve argument generation",
                    "falsifiable_criterion": "Tool argument error rate does not decrease after training with JSON examples",
                    "confidence": 0.80
                })
            
            elif failure_type == "wrong_tool_selection":
                hypotheses.append({
                    "hypothesis_id": f"hyp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_004",
                    "cluster_id": cluster.get("cluster_id"),
                    "failure_type": failure_type,
                    "statement": "Gemma fails to understand the prerequisite relationships between diagnostic and repair actions",
                    "evidence_references": cluster.get("example_trajectories", [])[:2],
                    "testable_prediction": "Training on sequences that show proper tool ordering will improve selection accuracy",
                    "falsifiable_criterion": "Wrong tool selection rate does not improve after sequential training",
                    "confidence": 0.82
                })
        
        # Limit total hypotheses
        hypotheses = hypotheses[:max_hypotheses_per_cluster * len(failure_clusters)]
        
        result = {
            "research_id": f"research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "failure_analysis_reference": fail_data.get("analysis_id", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "num_hypotheses_generated": len(hypotheses),
            "hypotheses": hypotheses,
            "research_summary": {
                "total_failure_clusters_analyzed": len(failure_clusters),
                "hypotheses_per_cluster": len(hypotheses) // max(len(failure_clusters), 1),
                "all_hypotheses_testable": True,
                "recommended_immediate_focus": "verification_behavior" if any(h["failure_type"] == "failed_verification" for h in hypotheses) else "tool_argument_formation"
            }
        }
        
        return json.dumps(result, indent=2)
    
    @tool
    def validate_hypothesis_testability(self, 
                                       hypotheses: str) -> str:
        """
        Validate that hypotheses are testable and falsifiable.
        
        Args:
            hypotheses: JSON string containing hypotheses to validate
            
        Returns:
            JSON string with validation results
        """
        try:
            hyp_data = json.loads(hypotheses)
            hyps = hyp_data.get("hypotheses", [])
        except (json.JSONDecodeError, KeyError):
            hyps = []
        
        validation_results = []
        for hyp in hyps:
            # Check testability criteria
            is_testable = (
                "testable_prediction" in hyp and 
                "falsifiable_criterion" in hyp and
                "evidence_references" in hyp and
                len(hyp.get("evidence_references", [])) > 0
            )
            
            validation_results.append({
                "hypothesis_id": hyp.get("hypothesis_id", "unknown"),
                "is_testable": is_testable,
                "has_evidence": len(hyp.get("evidence_references", [])) > 0,
                "has_prediction": "testable_prediction" in hyp,
                "has_falsifiable_criterion": "falsifiable_criterion" in hyp,
                "validation_notes": "Meets all testability criteria" if is_testable else "Missing required testability components"
            })
        
        return json.dumps({
            "validation_completed": True,
            "total_hypotheses_validated": len(validation_results),
            "validation_results": validation_results,
            "all_testable": all(v["is_testable"] for v in validation_results)
        }, indent=2)
    
    @tool
    def design_validation_experiment(self, 
                                    hypothesis: str,
                                    environment_config: Dict[str, Any]) -> str:
        """
        Design an experiment to validate a specific hypothesis.
        
        Args:
            hypothesis: JSON string containing hypothesis to validate
            environment_config: Configuration for the test environment
            
        Returns:
            JSON string describing the validation experiment
        """
        try:
            hyp_data = json.loads(hypothesis)
        except json.JSONDecodeError:
            hyp_data = {"raw_hypothesis": hypothesis}
        
        # In implementation, would design actual experiment
        experiment_design = {
            "experiment_id": f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "hypothesis_reference": hyp_data.get("hypothesis_id", "unknown"),
            "experiment_type": "behavioral_validation",
            "environment_setup": environment_config,
            "procedure": [
                "Create training dataset that exemplifies correct behavior per hypothesis",
                "Fine-tune Gemma on this dataset using QLoRA",
                "Run benchmark to measure if hypothesis-specific behavior improved",
                "Compare against control group (no specific training)",
                "Statistical significance testing"
            ],
            "success_metrics": hyp_data.get("testable_prediction", "Improved behavior"),
            "failure_metrics": hyp_data.get("falsifiable_criterion", "No improvement"),
            "required_resources": {
                "training_examples": "100-500",
                "estimated_training_time": "15-30 minutes",
                "evaluation_episodes": 20
            }
        }
        
        return json.dumps(experiment_design, indent=2)


# Factory function
def create_research_agent(model: str = None, model_provider: Any = None) -> ResearchAgent:
    """Create a Research Agent instance."""
    return ResearchAgent(model, model_provider)
