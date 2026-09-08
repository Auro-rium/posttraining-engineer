"""
Training Designer Agent - Chooses optimal QLoRA configurations within constraints.
Responsible for selecting training parameters that respect budget and resource limits.
"""
from strands import Agent, tool

from .prompt_contract import get_prompt_contract, resolve_agent_model
from strands.types.tools import ToolResult, ToolUse
from typing import Dict, Any, List
import json
from datetime import datetime
from app.core.state import OptimizationRun


class TrainingDesignerAgent:
    """Agent responsible for designing QLoRA training configurations."""
    
    def __init__(self, model: str = None, model_provider: Any = None):
        prompt_contract = get_prompt_contract("TrainingDesignerAgent")
        self.agent = Agent(
            name="TrainingDesignerAgent",
            model=resolve_agent_model(model, model_provider=model_provider),
            system_prompt=prompt_contract.prompt,
        )
        
        # Register tools
        self.prompt_contract = prompt_contract
        self.prompt_metadata = prompt_contract.metadata()
        self.agent.tool_registry.register_tool(self.analyze_dataset_characteristics)
        self.agent.tool_registry.register_tool(self.select_qlora_configuration)
        self.agent.tool_registry.register_tool(self.validate_configuration_constraints)
        self.agent.tool_registry.register_tool(self.estimate_training_resources)
    
    @tool
    def analyze_dataset_characteristics(self, 
                                       dataset_references: str) -> str:
        """
        Analyze the training dataset to inform configuration selection.
        
        Args:
            dataset_references: JSON string with SFT dataset S3 references
            
        Returns:
            JSON string with dataset analysis for training design
        """
        try:
            data_refs = json.loads(dataset_references)
            dataset_info = data_refs.get("dataset_references", [])
            stats = data_refs.get("dataset_statistics", {})
        except (json.JSONDecodeError, KeyError):
            dataset_info = []
            stats = {}
        
        # In implementation, would load actual dataset and analyze:
        # - Sequence lengths
        # - Token distribution  
        # - Complexity measures
        # - Diversity of examples
        
        # Simulate dataset analysis
        num_examples = stats.get("total_examples", 100)  # Default placeholder
        avg_length = stats.get("average_conversation_length", 4)
        
        analysis = {
            "dataset_analysis_id": f"data_anal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_references": dataset_info,
            "num_examples": num_examples,
            "average_sequence_length": avg_length,
            "estimated_tokens": num_examples * avg_length * 50,  # Rough estimate
            "complexity_assessment": "medium" if num_examples < 500 else "high",
            "diversity_assessment": "good" if num_examples > 50 else "limited",
            "recommended_training_approach": "standard_qlora",
            "warnings": [] if num_examples >= 20 else ["Dataset may be too small for effective training"],
            "constraints_informed_by_data": {
                "max_reasonable_epochs": min(4, max(1, 1000 // max(num_examples, 1))),
                "suggested_rank_range": [16, 32] if num_examples > 100 else [8, 16],
                "learning_rate_sensitivity": "medium"
            }
        }
        
        return json.dumps(analysis, indent=2)
    
    @tool
    def select_qlora_configuration(self, 
                                  dataset_analysis: str,
                                  budget_constraints: Dict[str, Any],
                                  allowed_configs: Dict[str, List[Any]] = None) -> str:
        """
        Select optimal QLoRA configuration within constraints.
        
        Args:
            dataset_analysis: JSON string from analyze_dataset_characteristics
            budget_constraints: Budget limits from optimization run
            allowed_configs: Dictionary of allowed parameter values (uses defaults if None)
            
        Returns:
            JSON string with selected QLoRA configuration and rationale
        """
        # Default allowed configurations (from spec)
        if allowed_configs is None:
            allowed_configs = {
                "rank": [8, 16, 32, 64],
                "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3],
                "epochs": [1, 2, 3, 4],
                "dropout": [0.05, 0.1, 0.15]
            }
        
        try:
            data_analysis = json.loads(dataset_analysis)
        except json.JSONDecodeError:
            data_analysis = {"raw_data": dataset_analysis}
        
        # Extract dataset info
        num_examples = data_analysis.get("num_examples", 100)
        complexity = data_analysis.get("complexity_assessment", "medium")
        warnings = data_analysis.get("warnings", [])
        
        # Extract budget constraints
        max_time = budget_constraints.get("maxTrainingTimeMin", 60)
        max_cost = budget_constraints.get("maxCostUSD", 20)
        
        # Simple selection algorithm - in reality would be more sophisticated
        # Select middle-range values that balance performance and constraints
        
        # Rank selection based on dataset size
        if num_examples < 50:
            selected_rank = 8  # Lower rank for small datasets to prevent overfitting
        elif num_examples < 200:
            selected_rank = 16
        else:
            selected_rank = 32  # Higher rank for larger datasets
            
        # Ensure rank is in allowed list
        if selected_rank not in allowed_configs["rank"]:
            selected_rank = min(allowed_configs["rank"], key=lambda x: abs(x - selected_rank))
        
        # Learning rate selection based on complexity and time
        if max_time < 20:  # Very limited time
            selected_lr = 5e-4  # Higher LR for faster convergence
        elif complexity == "high":
            selected_lr = 1e-4  # Lower LR for complex scenarios
        else:
            selected_lr = 2e-4  # Middle ground
            
        # Ensure LR is in allowed list
        if selected_lr not in allowed_configs["learning_rate"]:
            selected_lr = min(allowed_configs["learning_rate"], key=lambda x: abs(x - selected_lr))
        
        # Epochs selection based on time and dataset size
        time_per_epoch_estimate = max(5, num_examples // 25)  # Rough estimate
        max_epochs_by_time = max(1, max_time // time_per_epoch_estimate)
        selected_epochs = min(4, max(1, max_epochs_by_time))
        
        # Ensure epochs is in allowed list
        if selected_epochs not in allowed_configs["epochs"]:
            selected_epochs = min(allowed_configs["epochs"], key=lambda x: abs(x - selected_epochs))
        
        # Dropout selection - conservative choice
        selected_dropout = 0.1  # Middle value, good general choice
        if selected_dropout not in allowed_configs["dropout"]:
            selected_dropout = min(allowed_configs["dropout"], key=lambda x: abs(x - selected_dropout))
        
        # Build configuration
        selected_config = {
            "rank": selected_rank,
            "learning_rate": selected_lr,
            "epochs": selected_epochs,
            "dropout": selected_dropout
        }
        
        # Calculate estimated resource usage
        estimated_time = selected_epochs * time_per_epoch_estimate
        estimated_cost = estimated_time * 0.5  # Rough $0.50 per minute estimate
        
        # Validate against constraints
        time_ok = estimated_time <= max_time
        cost_ok = estimated_cost <= max_cost
        
        # Generate rationale
        rationale_parts = [
            f"Selected rank {selected_rank} based on dataset size ({num_examples} examples)",
            f"Selected learning rate {selected_lr} for {complexity} complexity scenario",
            f"Selected {selected_epochs} epochs to fit within {max_time} minute time budget",
            f"Selected dropout {selected_dropout} as reasonable regularization"
        ]
        
        if not time_ok:
            rationale_parts.append(f"WARNING: Estimated time ({estimated_time}min) exceeds budget ({max_time}min)")
        if not cost_ok:
            rationale_parts.append(f"WARNING: Estimated cost (${estimated_cost:.2f}) exceeds budget (${max_cost})")
        
        result = {
            "design_id": f"design_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_analysis_reference": data_analysis.get("dataset_analysis_id", "unknown"),
            "budget_constraints": budget_constraints,
            "allowed_configuration_space": allowed_configs,
            "selected_qlora_configuration": selected_config,
            "estimated_resource_usage": {
                "estimated_training_time_min": round(estimated_time, 1),
                "estimated_cost_usd": round(estimated_cost, 2),
                "estimated_total_tokens": data_analysis.get("estimated_tokens", 0)
            },
            "constraint_validation": {
                "within_time_budget": time_ok,
                "within_cost_budget": cost_ok,
                "all_parameters_allowed": True,  # By construction
                "validation_passed": time_ok and cost_ok
            },
            "selection_rationale": ". ".join(rationale_parts),
            "alternatives_considered": [
                {
                    "parameter": "rank",
                    "considered": allowed_configs["rank"],
                    "selected": selected_rank
                },
                {
                    "parameter": "learning_rate", 
                    "considered": allowed_configs["learning_rate"],
                    "selected": selected_lr
                },
                {
                    "parameter": "epochs",
                    "considered": allowed_configs["epochs"],
                    "selected": selected_epochs
                }
            ],
            "design_summary": {
                "configuration_centered": True,
                "constraint_aware": True,
                "optimization_within_bounds": time_ok and cost_ok,
                "ready_for_training_execution": time_ok and cost_ok
            }
        }
        
        return json.dumps(result, indent=2)
    
    @tool
    def validate_configuration_constraints(self, 
                                          configuration: str) -> str:
        """
        Validate that a configuration respects all constraints.
        
        Args:
            configuration: JSON string with QLoRA configuration to validate
            
        Returns:
            JSON string with validation results
        """
        try:
            config = json.loads(configuration)
        except json.JSONDecodeError:
            return json.dumps({
                "validation_failed": True,
                "error": "Invalid JSON configuration"
            }, indent=2)
        
        # Default allowed values (would come from system config in practice)
        allowed = {
            "rank": [8, 16, 32, 64],
            "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3],
            "epochs": [1, 2, 3, 4],
            "dropout": [0.05, 0.1, 0.15]
        }
        
        validation_errors = []
        warnings = []
        
        # Check each parameter
        for param_name, allowed_values in allowed.items():
            if param_name not in config:
                validation_errors.append(f"Missing required parameter: {param_name}")
            elif config[param_name] not in allowed_values:
                validation_errors.append(
                    f"Parameter {param_name}={config[param_name]} not in allowed values {allowed_values}"
                )
        
        # Additional sanity checks
        if config.get("learning_rate", 0) <= 0:
            validation_errors.append("Learning rate must be positive")
        if config.get("epochs", 0) < 1:
            validation_errors.append("Epochs must be at least 1")
        if config.get("rank", 0) < 1:
            validation_errors.append("Rank must be at least 1")
        if not (0 <= config.get("dropout", 0) <= 1):
            validation_errors.append("Dropout must be between 0 and 1")
        
        # Performance warnings
        if config.get("rank", 0) > 64:
            warnings.append("Very high rank may lead to overfitting")
        if config.get("learning_rate", 0) > 1e-3:
            warnings.append("Very high learning rate may cause instability")
        if config.get("epochs", 0) > 4:
            warnings.append("More than 4 epochs may lead to overfitting")
        
        is_valid = len(validation_errors) == 0
        
        result = {
            "configuration_validation_id": f"val_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "configuration_provided": config,
            "is_valid": is_valid,
            "validation_errors": validation_errors,
            "warnings": warnings,
            "allowed_values_reference": allowed,
            "validation_summary": {
                "all_parameters_recognized": True,
                "constraint_compliant": is_valid,
                "safe_to_use": is_valid and len(warnings) == 0
            }
        }
        
        return json.dumps(result, indent=2)
    
    @tool
    def estimate_training_resources(self, 
                                   configuration: str,
                                   dataset_info: str) -> str:
        """
        Estimate computational resources needed for training.
        
        Args:
            configuration: JSON string with QLoRA configuration
            dataset_info: JSON string with dataset information
            
        Returns:
            JSON string with resource estimation
        """
        try:
            config = json.loads(configuration)
            data_info = json.loads(dataset_info)
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid input JSON"
            }, indent=2)
        
        # Extract parameters
        rank = config.get("rank", 16)
        epochs = config.get("epochs", 2)
        # Learning rate and dropout have minimal direct impact on compute time
        
        # Extract dataset info
        num_examples = data_info.get("num_examples", 100)
        avg_length = data_info.get("average_sequence_length", 4)
        estimated_tokens = data_info.get("estimated_tokens", num_examples * avg_length * 50)
        
        # Resource estimation formulas (simplified)
        # Base time per example increases with rank and sequence length
        base_time_per_example = 0.1  # seconds
        rank_factor = rank / 16.0  # Normalize to rank=16
        length_factor = avg_length / 4.0  # Normalize to length=4
        
        time_per_example = base_time_per_example * rank_factor * length_factor
        total_time_seconds = estimated_tokens * time_per_example / 50  # Rough token processing
        total_time_minutes = total_time_seconds / 60
        
        # Add overhead for setup, saving, etc.
        overhead_minutes = 5  # Fixed overhead
        estimated_total_time = total_time_minutes + overhead_minutes
        
        # Cost estimation (rough AWS estimates)
        compute_cost_per_minute = 0.50  # $/hour for ml.g5.xlarge equivalent
        storage_cost = 0.01 * (estimated_tokens / 1000)  # Per 1K tokens storage
        estimated_cost = (estimated_total_time * compute_cost_per_minute) + storage_cost
        
        # Memory estimation (VRAM)
        base_vram = 2  # GB for base model
        rank_vram = rank * 0.02  # Approximate VRAM increase per rank unit
        estimated_vram = base_vram + rank_vram
        
        result = {
            "resource_estimation_id": f"res_est_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "configuration_used": config,
            "dataset_info_used": data_info,
            "resource_estimates": {
                "estimated_training_time_minutes": round(estimated_total_time, 1),
                "estimated_vram_gb": round(estimated_vram, 1),
                "estimated_storage_mb": round(estimated_storage_mb, 1) if 'estimated_storage_mb' in locals() else round(storage_cost * 10, 1),
                "estimated_total_tokens": estimated_tokens
            },
            "cost_estimate": {
                "estimated_cost_usd": round(estimated_cost, 2),
                "compute_cost_component": round(estimated_total_time * compute_cost_per_minute, 2),
                "storage_cost_component": round(storage_cost, 3)
            },
            "resource_summary": {
                "estimation_method": "parameter_based_scaling",
                "assumptions": [
                    "Linear scaling with rank and sequence length",
                    "Fixed overhead for setup/teardown",
                    "Standard AWS compute pricing assumptions"
                ],
                "confidence": "medium",  # Would be higher with profiling data
                "ready_for_budget_check": True
            }
        }
        
        return json.dumps(result, indent=2)


# Factory function
def create_training_designer_agent(model: str = None, model_provider: Any = None) -> TrainingDesignerAgent:
    """Create a Training Designer Agent instance."""
    return TrainingDesignerAgent(model, model_provider)
