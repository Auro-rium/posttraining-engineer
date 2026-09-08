"""Lazy Bedrock model construction for Strands agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class OptionalDependencyError(RuntimeError):
    """Raised when Strands is not installed and a live model is requested."""


class BedrockStrandsModel:
    """A small dependency-injected factory around ``strands.models.BedrockModel``.

    Importing this module and constructing the wrapper never creates a boto3
    session or contacts Bedrock.  Applications can pass ``model_factory`` in
    tests or use the default lazy Strands factory in AWS.
    """

    def __init__(
        self,
        model_id: str,
        *,
        region_name: str | None = None,
        boto_session: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        self.model_id = model_id
        self.region_name = region_name
        self.boto_session = boto_session
        self.model_factory = model_factory
        self.model_config = dict(model_config or {})
        self._model: Any | None = None

    def _factory(self) -> Callable[..., Any]:
        if self.model_factory is not None:
            return self.model_factory
        try:
            from strands.models import BedrockModel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise OptionalDependencyError(
                "Install strands-agents to use BedrockStrandsModel"
            ) from exc
        return BedrockModel

    def create_model(self) -> Any:
        if self._model is not None:
            return self._model
        kwargs = dict(self.model_config)
        kwargs.setdefault("model_id", self.model_id)
        if self.region_name is not None:
            kwargs.setdefault("region_name", self.region_name)
        if self.boto_session is not None:
            kwargs.setdefault("boto_session", self.boto_session)
        self._model = self._factory()(**kwargs)
        return self._model

    @property
    def model(self) -> Any:
        return self.create_model()

    def create_agent(
        self,
        *,
        name: str,
        system_prompt: str,
        tools: list[Any] | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        try:
            from strands import Agent
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise OptionalDependencyError(
                "Install strands-agents to create a Strands agent"
            ) from exc
        kwargs: dict[str, Any] = {
            "model": self.create_model(),
            "name": name,
            "system_prompt": system_prompt,
        }
        if tools is not None:
            kwargs["tools"] = tools
        kwargs.update(agent_kwargs)
        return Agent(**kwargs)

    def invoke(
        self,
        prompt: str,
        *,
        agent_name: str = "PostTrainingAgent",
        system_prompt: str = "",
        **kwargs: Any,
    ) -> Any:
        """Invoke a one-shot Strands agent using the lazily-created model."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        agent = self.create_agent(name=agent_name, system_prompt=system_prompt, **kwargs)
        return agent(prompt)


BedrockModelProvider = BedrockStrandsModel
