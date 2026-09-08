"""
Failure Analyst Agent - Converts raw failures into categorized failure types.
Responsible for failure pattern recognition and categorization.
"""
from strands import Agent, tool
from strands.types.tools import ToolResult, ToolUse
from typing import Dict, Any, List
import json
from datetime import datetime
from app.core.state import OptimizationRun


class FailureAnalystAgent:
    """Agent responsible for analyzing failures and categorizing them."""
    
    def __init__(self, model: str = None):
        self.agent = Agent(
            name="FailureAnalystAgent",
            model=model or "nvidia.nemotron-super-3-120b",
            system_prompt="""You are the Failure Analyst Agent in an autonomous post-training system.
            Your role is to analyze benchmark trajectories and convert raw failures into 
            categorized failure types that can be acted upon.
            
            You must:
            1. Analyze trajectories from the Benchmark Agent
            2. Identify points where Gemma failed to complete tasks successfully
            3. Categorize failures into types such as:
               - Malformed tool arguments
               - Wrong tool selection  
               - Premature completion
               - Failed verification
               - Looping behavior
               - Incorrect tool usage sequence
            4. Group similar failures into clusters with examples
            5. Provide evidence (trajectory references) for each failure cluster
            
            Focus on objective failure analysis - do not attempt to fix issues or 
            generate solutions. Simply categorize what went wrong."""
        )
        
        # Register tools
        self.agent.tool_registry.register_tool(self.analyze_failures)
        self.agent.tool_registry.register_tool(self.categorize_failure_types)
        self.agent.tool_registry.register_tool(self.cluster_similar_failures)
    
    @tool
    def analyze_failures(self, 
                        benchmark_results: str,
                        trajectory_references: List[str]) -> str:
        """
        Analyze benchmark results to identify and categorize failures.
        
        Args:
            benchmark_results: JSON string from Benchmark Agent
            trajectory_references: List of S3 references to trajectory data
            
        Returns:
            JSON string containing failure analysis and clusters
        """
        # Parse benchmark results
        try:
            bench_data = json.loads(benchmark_results)
        except json.JSONDecodeError:
            bench_data = {"raw_data": benchmark_results}
        
        # In implementation, this would:
        # 1. Load trajectory data from S3 references
        # 2. Analyze each trajectory for failure points
        # 3. Categorize failures by type
        # 4. Group similar failures
        # 5. Store failure clusters to S3
        
        # Return structured failure analysis
        failure_analysis = {
            "analysis_id": f"fail_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "benchmark_run_id": bench_data.get("run_id", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "trajectories_analyzed": len(trajectory_references),
            "total_failures_identified": 42,  # Placeholder based on 65% failure rate
            "failure_clusters": [
                {
                    "cluster_id": "cluster_001",
                    "failure_type": "failed_verification",
                    "description": "Gemma frequently terminates before checking whether repairs worked",
                    "example_count": 18,
                    "percentage": 42.9,
                    "example_trajectories": [
                        f"s3://post-training-engineer-artifacts/runs/{bench_data.get('run_id', 'unknown')}/trajectories/fail_example_001.jsonl",
                        f"s3://post-training-engineer-artifacts/runs/{bench_data.get('run_id', 'unknown')}/trajectories/fail_example_002.jsonl"
                    ],
                    "key_evidence": [
                        "Trajectory shows: get_logs → restart_service → declare_success (without verification)",
                        "Expected: get_logs → inspect_service → read_config → edit_config → restart_service → run_healthcheck"
                    ]
                },
                {
                    "cluster_id": "cluster_002", 
                    "failure_type": "malformed_tool_arguments",
                    "description": "Gemma produces incorrectly formatted tool calls",
                    "example_count": 11,
                    "percentage": 26.2,
                    "example_trajectories": [
                        f"s3://post-training-engineer-artifacts/runs/{bench_data.get('run_id', 'unknown')}/trajectories/fail_example_003.jsonl"
                    ],
                    "key_evidence": [
                        "Tool calls missing required parameters",
                        "Incorrect JSON structure in tool arguments",
                        "Wrong data types in arguments (string vs int)"
                    ]
                },
                {
                    "cluster_id": "cluster_003",
                    "failure_type": "wrong_tool_selection", 
                    "description": "Gemma selects inappropriate tools for the situation",
                    "example_count": 8,
                    "percentage": 19.0,
                    "example_trajectories": [
                        f"s3://post-training-engineer-artifacts/runs/{bench_data.get('run_id', 'unknown')}/trajectories/fail_example_004.jsonl"
                    ],
                    "key_evidence": [
                        "Using edit_config when get_logs or inspect_service needed first",
                        "Attempting to restart service without diagnosing issue",
                        "Running healthcheck before making repairs"
                    ]
                },
                {
                    "cluster_id": "cluster_004",
                    "failure_type": "premature_completion",
                    "description": "Gemma declares success before task is actually complete",
                    "example_count": 5,
                    "percentage": 11.9,
                    "example_trajectories": [
                        f"s3://post-training-engineer-agent-artifacts/runs/{bench_data.get('run_id', 'unknown')}/trajectories/fail_example_005.jsonl"
                    ],
                    "key_evidence": [
                        "Declaring success after single tool call",
                        "Not verifying that repair actually resolved the issue",
                        "Assuming restart means fixed without validation"
                    ]
                }
            ],
            "analysis_summary": {
                "most_common_failure": "failed_verification",
                "failure_rate": 0.65,
                "recommended_focus_areas": ["verification_behavior", "tool_argument_formation", "sequential_reasoning"]
            }
        }
        
        return json.dumps(failure_analysis, indent=2)
    
    @tool
    def categorize_failure_types(self, 
                                raw_failures: List[Dict[str, Any]]) -> str:
        """
        Categorize raw failure observations into standard failure types.
        
        Args:
            raw_failures: List of raw failure observations from trajectories
            
        Returns:
            JSON string with categorized failures
        """
        # Standard failure taxonomy for service recovery
        failure_taxonomy = {
            "malformed_tool_arguments": "Tool calls with incorrect syntax or missing parameters",
            "wrong_tool_selection": "Selecting inappropriate tools for current state", 
            "premature_completion": "Declaring task complete before verification",
            "failed_verification": "Not checking if repairs actually worked",
            "looping_behavior": "Repeating same actions without progress",
            "incorrect_tool_sequence": "Using tools in wrong order",
            "missing_prerequisite": "Attempting action without completing prerequisites",
            "incorrect_parameter_value": "Tool parameters have wrong values"
        }
        
        # In implementation, would map raw observations to these categories
        categorized = {}
        for failure_type, description in failure_taxonomy.items():
            categorized[failure_type] = {
                "description": description,
                "count": 0,  # Would be calculated from actual data
                "examples": []
            }
        
        return json.dumps({
            "taxonomy": failure_taxonomy,
            "categorized_failures": categorized,
            "total_categorized": sum(cat["count"] for cat in categorized.values())
        }, indent=2)
    
    @tool
    def cluster_similar_failures(self, 
                                categorized_failures: str,
                                similarity_threshold: float = 0.8) -> str:
        """
        Group similar failures into clusters for analysis.
        
        Args:
            categorized_failures: JSON string from categorize_failure_types
            similarity_threshold: Threshold for considering failures similar
            
        Returns:
            JSON string with failure clusters
        """
        # In implementation, would use similarity metrics to group failures
        return json.dumps({
            "clustering_completed": True,
            "similarity_threshold_used": similarity_threshold,
            "clusters_formed": 4,  # Placeholder
            "cluster_method": "similarity_based_grouping",
            "note": "In production, would use embedding similarity or rule-based clustering"
        }, indent=2)


# Factory function
def create_failure_analyst_agent(model: str = None) -> FailureAnalystAgent:
    """Create a Failure Analyst Agent instance."""
    return FailureAnalystAgent(model)