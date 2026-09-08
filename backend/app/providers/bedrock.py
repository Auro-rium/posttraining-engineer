"""Lazy Bedrock model construction for Strands agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

BedrockAuthMode = Literal["sigv4"]
"""Supported Bedrock authentication mode.

The application deliberately supports AWS SigV4 only.  ``AWS_BEARER_TOKEN_BEDROCK``
is a separate bearer-token mechanism and, when set to a stale value, can make
otherwise valid IAM credentials fail.  Keeping the mode explicit prevents a
deployment from silently selecting that mechanism.
"""

SIGV4_AUTH_MODE: BedrockAuthMode = "sigv4"


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
        auth_mode: BedrockAuthMode = SIGV4_AUTH_MODE,
        model_factory: Callable[..., Any] | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if auth_mode != SIGV4_AUTH_MODE:
            raise ValueError("BedrockStrandsModel supports SigV4 authentication only")
        self.model_id = model_id
        self.region_name = region_name
        self.boto_session = boto_session
        self.auth_mode = auth_mode
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
        session = self.boto_session
        # Constructing an explicit boto3 session is the important auth
        # boundary: it uses the configured IAM credential chain and SigV4.
        # Do not pass bearer-token configuration to Strands.  Tests can still
        # inject a factory without importing boto3 or creating a session.
        if session is None and self.model_factory is None and self.auth_mode == SIGV4_AUTH_MODE:
            try:
                import boto3  # type: ignore[import-untyped]
                from botocore.config import Config  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise OptionalDependencyError("Install boto3 to use SigV4 Bedrock auth") from exc
            session = boto3.Session(region_name=self.region_name)
            # Bedrock advertises both SigV4 and bearer auth.  Botocore may
            # otherwise prefer AWS_BEARER_TOKEN_BEDROCK when it is present,
            # even if the application is configured for IAM credentials.
            # Explicitly selecting the SigV4 scheme makes the auth boundary
            # deterministic without mutating the caller's environment.
            kwargs.setdefault("boto_client_config", Config(signature_version="v4"))
        if session is not None:
            kwargs.setdefault("boto_session", session)
        elif self.region_name is not None:
            kwargs.setdefault("region_name", self.region_name)
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
