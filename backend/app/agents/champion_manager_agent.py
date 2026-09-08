"""
Champion Manager Agent - Makes deterministic promotion decisions based on gates.
Responsible for accept/reject decisions using predefined improvement and regression thresholds.
"""
from strands import Agent, tool

from .prompt_contract import get_prompt_contract, resolve_agent_model
from strands.types.tools import ToolResult, ToolUse
from typing import Dict, Any, List
import json
from datetime import datetime
from app.core.state import OptimizationRun


class ChampionManagerAgent:
    """Agent responsible for making promotion decisions."""
    
    def __init__(self, model: str = None, model_provider: Any = None):
        prompt_contract = get_prompt_contract("ChampionManagerAgent")
        self.agent = Agent(
            name="ChampionManagerAgent",
            model=resolve_agent_model(model, model_provider=model_provider),
            system_prompt=prompt_contract.prompt,
        )
        
        # Register tools
        self.prompt_contract = prompt_contract
        self.prompt_metadata = prompt_contract.metadata()
        self.agent.tool_registry.register_tool(self.apply_improvement_gate)
        self.agent.tool_registry.register_tool(self.apply_regression_gate)
        self.agent.tool_registry.register_tool(self.make_promotion_decision)
        self.agent.tool_registry.register_tool(self.update_champion_reference)
    
    @tool
    def apply_improvement_gate(self, 
                              candidate_performance: float,
                              champion_performance: float,
                              min_improvement: float = 0.10) -> str:
        """
        Apply the improvement gate: candidate must beat champion by minimum threshold.
        
        Args:
            candidate_performance: Performance score of candidate model
            champion_performance: Performance score of current champion
            min_improvement: Minimum relative improvement required (default 0.10 = 10%)
            
        Returns:
            JSON string with improvement gate evaluation
        """
        if champion_performance <= 0:
            # Handle edge case where champion performance is zero or negative
            relative_improvement = float('inf') if candidate_performance > 0 else 0
            improvement_threshold_met = candidate_performance > 0
        else:
            relative_improvement = (candidate_performance - champion_performance) / champion_performance
            improvement_threshold_met = relative_improvement >= min_improvement
        
        improvement_delta = candidate_performance - champion_performance
        
        result = {
            "gate_evaluation_id": f"improve_gate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "gate_type": "improvement",
            "candidate_performance": candidate_performance,
            "champion_performance": champion_performance,
            "absolute_improvement": round(improvement_delta, 4),
            "relative_improvement": round(relative_improvement, 4) if champion_performance != 0 else None,
            "improvement_threshold": min_improvement,
            "gate_passed": improvement_threshold_met,
            "gate_rationale": {
                "gate_description": "Candidate must show minimum relative improvement over champion",
                "calculation": f"({candidate_performance} - {champion_performance}) / {champion_performance} >= {min_improvement}" if champion_performance != 0 else f"Candidate performance {candidate_performance} must be > 0",
                "result": f"{relative_improvement:.4f} >= {min_improvement}" if champion_performance != 0 else f"{candidate_performance} > 0",
                "status": "PASS" if improvement_threshold_met else "FAIL"
            }
        }
        
        return json.dumps(result, indent=2)
    
    @tool
    def apply_regression_gate(self, 
                             candidate_performance: float,
                             champion_performance: float,
                             max_regression: float = 0.05) -> str:
        """
        Apply the regression gate: candidate must not regress beyond maximum threshold.
        
        Args:
            candidate_performance: Performance score of candidate model
            champion_performance: Performance score of current champion
            max_regression: Maximum allowed regression (default 0.05 = 5% absolute)
            
        Returns:
            JSON string with regression gate evaluation
        """
        # Regression is measured as champion - candidate (positive = regression)
        regression_delta = champion_performance - candidate_performance
        regression_threshold_met = regression_delta <= max_regression
        
        result = {
            "gate_evaluation_id": f"regress_gate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "gate_type": "regression",
            "candidate_performance": candidate_performance,
            "champion_performance": champion_performance,
            "regression_delta": round(regression_delta, 4),  # Positive = regression
            "regression_threshold": max_regression,
            "gate_passed": regression_threshold_met,
            "gate_rationale": {
                "gate_description": "Candidate must not exceed maximum allowed regression from champion",
                "calculation": f"{champion_performance} - {candidate_performance} <= {max_regression}",
                "result": f"{regression_delta:.4f} <= {max_regression}",
                "status": "PASS" if regression_threshold_met else "FAIL"
            }
        }
        
        return json.dumps(result, indent=2)
    
    @tool
    def make_promotion_decision(self, 
                               improvement_gate_result: str,
                               regression_gate_result: str,
                               candidate_info: str = None) -> str:
        """
        Make final promotion decision based on gate results.
        
        Args:
            improvement_gate_result: JSON string from apply_improvement_gate
            regression_gate_result: JSON string from apply_regression_gate
            candidate_info: Optional information about the candidate
            
        Returns:
            JSON string with final promotion decision
        """
        try:
            improve_data = json.loads(improvement_gate_result)
            regress_data = json.loads(regression_gate_result)
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid gate result JSON"
            }, indent=2)
        
        # Extract gate results
        improve_passed = improve_data.get("gate_passed", False)
        regress_passed = regress_data.get("gate_passed", False)
        
        # Extract performance values
        candidate_perf = improve_data.get("candidate_performance", 0.0)
        champion_perf = improve_data.get("champion_performance", 0.0)
        
        # Make decision: BOTH gates must pass for promotion
        promotion_approved = improve_passed and regress_passed
        
        # Determine specific failure reason if rejected
        if not promotion_approved:
            if not improve_passed and not regress_passed:
                failure_reason = "failed_both_improvement_and_regression_gates"
            elif not improve_passed:
                failure_reason = "failed_improvement_gate"
            else:  # not regress_passed
                failure_reason = "failed_regression_gate"
        else:
            failure_reason = None
        
        # Build decision result
        result = {
            "decision_id": f"promotion_decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "decision_type": "promotion_evaluation",
            "candidate_performance": candidate_perf,
            "champion_performance": champion_perf,
            "improvement_gate": {
                "passed": improve_passed,
                "details": improve_data.get("gate_rationale", {})
            },
            "regression_gate": {
                "passed": regress_passed,
                "details": regress_data.get("gate_rationale", {})
            },
            "promotion_decision": "PROMOTE" if promotion_approved else "REJECT",
            "decision_rationale": {
                "decision_rule": "BOTH improvement gate AND regression gate must PASS for promotion",
                "improvement_gate_status": "PASS" if improve_passed else "FAIL",
                "regression_gate_status": "PASS" if regress_passed else "FAIL",
                "final_decision": "PROMOTE" if promotion_approved else "REJECT",
                "reasoning": f"Improvement gate: {'PASS' if improve_passed else 'FAIL'}, Regression gate: {'PASS' if regress_passed else 'FAIL'}"
            },
            "failure_analysis": {
                "decision_rejected": not promotion_approved,
                "failure_reason": failure_reason,
                "candidate_performance_delta": round(candidate_perf - champion_perf, 4),
                "would_promote_if": {
                    "improvement_needed": max(0, (champion_perf * 0.1) - (candidate_perf - champion_perf)) if not improve_passed else 0,
                    "regression_tolerance": max(0, (candidate_perf - champion_perf) - 0.05) if not regress_passed else 0
                } if not promotion_approved else None
            }
        }
        
        # Add candidate info if provided
        if candidate_info:
            try:
                result["candidate_information"] = json.loads(candidate_info)
            except json.JSONDecodeError:
                result["candidate_information"] = {"raw_info": candidate_info}
        
        result["decision_summary"] = {
            "decision_made": True,
            "decision_deterministic": True,
            "judgment_overridden": False,  # Key point: no discretion used
            "gates_applied_as_configured": True,
            "ready_for_state_update": True,
            "champion_will_change": promotion_approved
        }
        
        return json.dumps(result, indent=2)
    
    @tool
    def update_champion_reference(self, 
                                 promotion_decision: str,
                                 candidate_artifact_ref: str,
                                 current_champion_ref: str = None) -> str:
        """
        Update the champion reference if promotion is approved.
        
        Args:
            promotion_decision: JSON string from make_promotion_decision
            candidate_artifact_ref: S3 reference to promoted candidate model
            current_champion_ref: Current champion reference (if any)
            
        Returns:
            JSON string with champion update information
        """
        try:
            decision = json.loads(promotion_decision)
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid promotion decision JSON"
            }, indent=2)
        
        decision_promoted = decision.get("promotion_decision") == "PROMOTE"
        
        result = {
            "update_id": f"champ_update_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "promotion_decision": decision.get("promotion_decision", "UNKNOWN"),
            "champion_updated": decision_promoted,
            "previous_champion_reference": current_champion_ref,
            "new_champion_reference": candidate_artifact_ref if decision_promoted else current_champion_ref,
            "update_performed": decision_promoted,
            "update_rationale": {
                "update_rule": "Champion reference updated ONLY when promotion decision is PROMOTE",
                "decision_was": decision.get("promotion_decision", "UNKNOWN"),
                "action_taken": "Reference updated to candidate" if decision_promoted else "Reference unchanged (candidate rejected)",
                "champion_changed": decision_promoted
            }
        }
        
        if decision_promoted:
            result["update_details"] = {
                "promoted_model_source": candidate_artifact_ref,
                "champion_change_type": "initial_champion" if not current_champion_ref else "champion_replacement",
                "update_reason": "Candidate passed both improvement and regression gates"
            }
        else:
            result["update_details"] = {
                "rejected_model_source": candidate_artifact_ref,
                "champion_remains": current_champion_ref,
                "update_reason": "Candidate failed one or more promotion gates"
            }
        
        result["update_summary"] = {
            "update_completed": True,
            "champion_reference_now_points_to": result["new_champion_reference"],
            "deterministic_update": True,
            "no_discretion_used": True,
            "audit_trail_available": True
        }
        
        return json.dumps(result, indent=2)


# Factory function
def create_champion_manager_agent(model: str = None, model_provider: Any = None) -> ChampionManagerAgent:
    """Create a Champion Manager Agent instance."""
    return ChampionManagerAgent(model, model_provider)
