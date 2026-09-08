"""
AgentGym Service Recovery Environment Implementation.
Simulates a service recovery scenario where agents must diagnose and fix service issues.
"""
import asyncio
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class ToolResult:
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class ServiceConfig:
    """Configuration for a service that can fail."""
    name: str
    dependencies: List[str]
    config_file: str
    health_check_endpoint: str
    failure_modes: List[str]


class ServiceRecoveryEnvironment:
    """
    AgentGym-compatible service recovery environment.
    Agents must use tools to diagnose and fix service issues.
    """

    def __init__(self, environment_id: str = "service-recovery-001"):
        self.environment_id = environment_id
        self.services = self._initialize_services()
        self.current_issues = []
        self.episode_step = 0
        self.max_steps = 10

    def _initialize_services(self) -> Dict[str, ServiceConfig]:
        """Initialize the services in our environment."""
        return {
            "web": ServiceConfig(
                name="web",
                dependencies=["api", "database"],
                config_file="/etc/nginx/nginx.conf",
                health_check_endpoint="/health",
                failure_modes=["config_error", "dependency_failure", "port_conflict"]
            ),
            "api": ServiceConfig(
                name="api",
                dependencies=["database", "cache"],
                config_file="/etc/api/settings.py",
                health_check_endpoint="/api/health",
                failure_modes=["config_error", "dependency_failure", "code_error"]
            ),
            "database": ServiceConfig(
                name="database",
                dependencies=[],
                config_file="/etc/postgresql/postgresql.conf",
                health_check_endpoint="/db/health",
                failure_modes=["config_error", "connection_limit", "disk_full"]
            ),
            "cache": ServiceConfig(
                name="cache",
                dependencies=[],
                config_file="/etc/redis/redis.conf",
                health_check_endpoint="/cache/health",
                failure_modes=["config_error", "memory_limit"]
            )
        }

    def reset(self) -> Dict[str, Any]:
        """Reset the environment to a new episode with random issues."""
        self.episode_step = 0
        self.current_issues = self._generate_random_issues()

        return self._get_observation()

    def _generate_random_issues(self) -> List[Dict[str, Any]]:
        """Generate random service issues for the episode."""
        issues = []
        num_issues = random.randint(1, 3)

        for _ in range(num_issues):
            service_name = random.choice(list(self.services.keys()))
            service = self.services[service_name]
            failure_mode = random.choice(service.failure_modes)

            issues.append({
                "service": service_name,
                "failure_mode": failure_mode,
                "description": f"{service_name} service has {failure_mode}",
                "requires_verification": True
            })

        return issues

    def _get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        return {
            "environment_id": self.environment_id,
            "step": self.episode_step,
            "max_steps": self.max_steps,
            "active_issues": self.current_issues.copy(),
            "available_tools": [
                "get_logs",
                "inspect_service",
                "read_config",
                "edit_config",
                "restart_service",
                "run_healthcheck"
            ]
        }

    async def step(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute a tool step in the environment.

        Returns:
            observation, reward, done, info
        """
        self.episode_step += 1

        # Execute the tool
        result = await self._execute_tool(tool_name, tool_args)

        # Calculate reward
        reward = self._calculate_reward(result)

        # Check if episode is done
        done = self._is_episode_done()

        # Get new observation
        observation = self._get_observation()

        # Additional info
        info = {
            "tool_used": tool_name,
            "tool_args": tool_args,
            "tool_result": result.to_dict() if hasattr(result, 'to_dict') else result,
            "step": self.episode_step
        }

        return observation, reward, done, info

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        """Execute a specific tool and return result."""
        if tool_name == "get_logs":
            return await self._get_logs(tool_args)
        elif tool_name == "inspect_service":
            return await self._inspect_service(tool_args)
        elif tool_name == "read_config":
            return await self._read_config(tool_args)
        elif tool_name == "edit_config":
            return await self._edit_config(tool_args)
        elif tool_name == "restart_service":
            return await self._restart_service(tool_args)
        elif tool_name == "run_healthcheck":
            return await self._run_healthcheck(tool_args)
        else:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )

    async def _get_logs(self, args: Dict[str, Any]) -> ToolResult:
        """Get logs for a service."""
        service_name = args.get("service")
        if not service_name or service_name not in self.services:
            return ToolResult(success=False, error=f"Invalid service: {service_name}")

        # Simulate getting logs - in reality this would call actual service
        service = self.services[service_name]

        # Check if service has issues
        issue_found = any(
            issue["service"] == service_name
            for issue in self.current_issues
        )

        if issue_found:
            # Return logs showing the problem
            log_messages = [
                f"[{service_name}] Starting service...",
                f"[{service_name}] Loading configuration from {service.config_file}",
                f"[{service_name}] ERROR: Detected {self._get_issue_description(service_name)}",
                f"[{service_name}] Service failed to start properly"
            ]
        else:
            log_messages = [
                f"[{service_name}] Starting service...",
                f"[{service_name}] Loading configuration from {service.config_file}",
                f"[{service_name}] Service started successfully",
                f"[{service_name}] Listening on port 80",
                f"[{service_name}] Health check passed"
            ]

        return ToolResult(
            success=True,
            data={
                "service": service_name,
                "logs": "\n".join(log_messages),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def _inspect_service(self, args: Dict[str, Any]) -> ToolResult:
        """Inspect a service's current state and dependencies."""
        service_name = args.get("service")
        if not service_name or service_name not in self.services:
            return ToolResult(success=False, error=f"Invalid service: {service_name}")

        service = self.services[service_name]
        issue_found = any(
            issue["service"] == service_name
            for issue in self.current_issues
        )

        inspection_data = {
            "service": service_name,
            "status": ServiceStatus.DOWN.value if issue_found else ServiceStatus.HEALTHY.value,
            "dependencies": service.dependencies,
            "config_file": service.config_file,
            "health_endpoint": service.health_check_endpoint,
            "issues": [
                issue for issue in self.current_issues
                if issue["service"] == service_name
            ]
        }

        return ToolResult(success=True, data=inspection_data)

    async def _read_config(self, args: Dict[str, Any]) -> ToolResult:
        """Read a service's configuration file."""
        service_name = args.get("service")
        if not service_name or service_name not in self.services:
            return ToolResult(success=False, error=f"Invalid service: {service_name}")

        service = self.services[service_name]

        # Simulate reading config - would actually read file
        issue_found = any(
            issue["service"] == service_name
            for issue in self.current_issues
        )

        if issue_found:
            # Return config with error
            config_content = f"""# {service.name} configuration
# WARNING: Configuration contains error that causes {self._get_current_failure_mode(service_name)}
# This needs to be fixed manually
setting1=value1
setting2=broken_value_should_be_fixed
setting3=value3
"""
        else:
            config_content = f"""# {service.name} configuration
# Healthy configuration
setting1=value1
setting2=value2
setting3=value3
"""

        return ToolResult(
            success=True,
            data={
                "service": service_name,
                "config_file": service.config_file,
                "content": config_content
            }
        )

    async def _edit_config(self, args: Dict[str, Any]) -> ToolResult:
        """Edit a service's configuration file."""
        service_name = args.get("service")
        content = args.get("content")

        if not service_name or service_name not in self.services:
            return ToolResult(success=False, error=f"Invalid service: {service_name}")
        if content is None:
            return ToolResult(success=False, error="Content is required for edit_config")

        # Simulate editing config - would actually write file
        service = self.services[service_name]

        # Check if the edit fixes the issue
        issue_fixed = self._validate_config_edit(service_name, content)

        if issue_fixed:
            # Remove the fixed issue
            self.current_issues = [
                issue for issue in self.current_issues
                if issue["service"] != service_name
            ]

            return ToolResult(
                success=True,
                data={
                    "service": service_name,
                    "config_file": service.config_file,
                    "message": f"Configuration updated successfully for {service_name}",
                    "issue_fixed": True
                }
            )
        else:
            return ToolResult(
                success=False,
                error="Configuration edit did not resolve the issue",
                data={
                    "service": service_name,
                    "config_file": service.config_file,
                    "issue_fixed": False
                }
            )

    async def _restart_service(self, args: Dict[str, Any]) -> ToolResult:
        """Restart a service."""
        service_name = args.get("service")
        if not service_name or service_name not in self.services:
            return ToolResult(success=False, error=f"Invalid service: {service_name}")

        service = self.services[service_name]
        issue_found = any(
            issue["service"] == service_name
            for issue in self.current_issues
        )

        if issue_found:
            # Service still has issues after restart
            return ToolResult(
                success=False,
                data={
                    "service": service_name,
                    "message": f"Service {service_name} restarted but still has issues",
                    "issue_persists": True
                }
            )
        else:
            # Service is healthy
            return ToolResult(
                success=True,
                data={
                    "service": service_name,
                    "message": f"Service {service_name} restarted successfully",
                    "issue_persists": False
                }
            )

    async def _run_healthcheck(self, args: Dict[str, Any]) -> ToolResult:
        """Run health check on service or system."""
        service_name = args.get("service")

        if service_name:
            # Check specific service
            if service_name not in self.services:
                return ToolResult(success=False, error=f"Invalid service: {service_name}")

            issue_found = any(
                issue["service"] == service_name
                for issue in self.current_issues
            )

            if issue_found:
                return ToolResult(
                    success=False,
                    data={
                        "service": service_name,
                        "healthy": False,
                        "message": f"Service {service_name} health check failed",
                        "issues": [
                            issue for issue in self.current_issues
                            if issue["service"] == service_name
                        ]
                    }
                )
            else:
                return ToolResult(
                    success=True,
                    data={
                        "service": service_name,
                        "healthy": True,
                        "message": f"Service {service_name} health check passed"
                    }
                )
        else:
            # Check overall system health
            all_healthy = len(self.current_issues) == 0

            return ToolResult(
                success=all_healthy,
                data={
                    "healthy": all_healthy,
                    "message": "All services healthy" if all_healthy else "Some services have issues",
                    "active_issues": len(self.current_issues),
                    "issues": self.current_issues.copy()
                }
            )

    def _get_issue_description(self, service_name: str) -> str:
        """Get description of current issue for a service."""
        for issue in self.current_issues:
            if issue["service"] == service_name:
                return issue["failure_mode"]
        return "unknown issue"

    def _get_current_failure_mode(self, service_name: str) -> str:
        """Get current failure mode for a service."""
        for issue in self.current_issues:
            if issue["service"] == service_name:
                return issue["failure_mode"]
        return "none"

    def _validate_config_edit(self, service_name: str, content: str) -> bool:
        """Validate if a config edit resolves the service issue."""
        # Simple validation - in reality would be more complex
        issue = next(
            (issue for issue in self.current_issues if issue["service"] == service_name),
            None
        )

        if not issue:
            return True  # No issue to fix

        failure_mode = issue["failure_mode"]

        # Simple heuristics for common failure modes
        if failure_mode == "config_error":
            # Check if config looks reasonable (not obviously broken)
            return "broken_value_should_be_fixed" not in content
        elif failure_mode == "dependency_failure":
            # Check if dependencies are mentioned/configured
            service = self.services[service_name]
            return all(dep in content for dep in service.dependencies)
        else:
            # For other modes, assume generic validation
            return len(content.strip()) > 10  # Not empty

    def _is_episode_done(self) -> bool:
        """Check if the episode is complete."""
        # Done if no issues remain or max steps reached
        return len(self.current_issues) == 0 or self.episode_step >= self.max_steps

    def _calculate_reward(self, result: ToolResult) -> float:
        """Calculate reward for the step."""
        # Base reward for taking action
        reward = 0.0

        if hasattr(result, 'success'):
            if result.success:
                reward += 0.1  # Small reward for successful action

                # Extra reward if action resolved issues
                if hasattr(result, 'data') and isinstance(result.data, dict):
                    if result.data.get('issue_fixed') is True:
                        reward += 1.0  # Big reward for fixing issue
                    elif result.data.get('healthy') is True:
                        reward += 0.5  # Medium reward for healthy state
            else:
                reward -= 0.05  # Small penalty for failed action

        # Big reward for solving all issues
        if len(self.current_issues) == 0 and self.episode_step > 0:
            reward += 2.0  # Episode completion bonus

        # Small penalty for each step to encourage efficiency
        reward -= 0.01 * self.episode_step

        return max(-1.0, min(3.0, reward))  # Clamp reward


# Factory function for easy instantiation
def create_service_recovery_environment(environment_id: str = None) -> ServiceRecoveryEnvironment:
    """Create a new service recovery environment."""
    return ServiceRecoveryEnvironment(environment_id or f"service-recovery-{random.randint(1000, 9999)}")
