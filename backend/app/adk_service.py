"""Typed A2A entrypoint for Cloud Run research and execution services.

Run with: ``uvicorn app.adk_service:a2a_app --host 0.0.0.0 --port 8080``.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.applications import Starlette

from app.adk_agents import build_service_agent
from app.cloud_provider import (
    A2AOperationRequest,
    GoogleIdentityTokenProvider,
)
from app.service_operations import (
    A2AOperationService,
    ADKStructuredResearchGenerator,
    GCSGroundedRetriever,
    GCSResearchEvidenceLoader,
    GroundedRetriever,
    ObjectiveEvidenceExecutor,
    RemoteObjectiveEvidenceExecutor,
    ResearchEvidenceLoader,
    ServiceOperationError,
    StructuredResearchGenerator,
)
from app.settings import Settings, get_settings


def _operation_request_from_message(message: Any) -> A2AOperationRequest:
    """Read exactly one protobuf data part and validate the wire request."""

    try:
        from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - cloud dependency only
        raise ServiceOperationError("PROTOBUF_UNAVAILABLE", "protobuf is unavailable") from exc
    if message is None or len(message.parts) != 1:
        raise ServiceOperationError(
            "INVALID_REQUEST", "A2A request must contain exactly one data part"
        )
    part = message.parts[0]
    if not part.HasField("data"):
        raise ServiceOperationError("INVALID_REQUEST", "A2A request part must be typed data")
    try:
        return A2AOperationRequest.model_validate(MessageToDict(part.data))
    except Exception as exc:
        raise ServiceOperationError(
            "INVALID_REQUEST", "A2A operation request failed schema validation"
        ) from exc


def _data_part(value: dict[str, Any]) -> Any:
    """Encode a JSON object as the v1 A2A protobuf data part."""

    try:
        from a2a.types import Part
        from google.protobuf.json_format import ParseDict
    except ImportError as exc:  # pragma: no cover - cloud dependency only
        raise ServiceOperationError("A2A_UNAVAILABLE", "A2A dependencies are unavailable") from exc
    part = Part()
    ParseDict(value, part.data)
    return part


def build_operation_agent_executor(service: A2AOperationService) -> Any:
    """Build the official A2A ``AgentExecutor`` bridge for typed operations."""

    try:
        from a2a.server.agent_execution import AgentExecutor
        from a2a.server.tasks import TaskUpdater
        from a2a.types import Message, Part, Role
    except ImportError as exc:  # pragma: no cover - cloud dependency only
        raise RuntimeError("install the cloud dependency extra to serve A2A") from exc

    class TypedOperationAgentExecutor(AgentExecutor):
        async def execute(self, context: Any, event_queue: Any) -> None:
            task_id = context.task_id or f"task_{uuid4().hex}"
            context_id = context.context_id or f"context_{uuid4().hex}"
            updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
            )
            await updater.submit()
            await updater.start_work()
            try:
                request = _operation_request_from_message(context.message)
                response = await service.handle(request)
                await updater.add_artifact(
                    parts=[_data_part(response.model_dump(mode="json"))],
                    name=f"{response.operation.value}.json",
                    last_chunk=True,
                )
                await updater.complete()
            except ServiceOperationError as exc:
                await updater.failed(
                    Message(
                        message_id=f"message_{uuid4().hex}",
                        role=Role.ROLE_AGENT,
                        parts=[Part(text=f"{exc.code}: operation failed closed")],
                    )
                )
            except Exception:
                await updater.failed(
                    Message(
                        message_id=f"message_{uuid4().hex}",
                        role=Role.ROLE_AGENT,
                        parts=[Part(text="INTERNAL_ERROR: operation failed closed")],
                    )
                )

        async def cancel(self, context: Any, event_queue: Any) -> None:
            updater = TaskUpdater(
                event_queue=event_queue,
                task_id=context.task_id or f"task_{uuid4().hex}",
                context_id=context.context_id or f"context_{uuid4().hex}",
            )
            await updater.cancel()

    return TypedOperationAgentExecutor()


def _build_operation_service(
    settings: Settings,
    *,
    research_generator: StructuredResearchGenerator | None = None,
    evidence_loader: ResearchEvidenceLoader | None = None,
    retriever: GroundedRetriever | None = None,
    objective_executor: ObjectiveEvidenceExecutor | None = None,
) -> A2AOperationService:
    if settings.service_role not in {"research", "execution"}:
        raise ValueError("A2A service is available only for research or execution roles")
    if objective_executor is None:
        if not settings.objective_execution_url:
            raise ValueError("OBJECTIVE_EXECUTION_URL is required for A2A services")
        objective_executor = RemoteObjectiveEvidenceExecutor(
            service_url=settings.objective_execution_url,
            token_provider=GoogleIdentityTokenProvider(),
            timeout_seconds=settings.a2a_timeout_seconds,
        )
    if settings.service_role == "research":
        research_generator = research_generator or ADKStructuredResearchGenerator(
            model=settings.gemini_model
        )
        if evidence_loader is None:
            if not settings.artifact_bucket:
                raise ValueError("ARTIFACT_BUCKET is required for research evidence")
            evidence_loader = GCSResearchEvidenceLoader(
                bucket=settings.artifact_bucket,
                project=settings.google_cloud_project,
            )
        if retriever is None:
            if not settings.rag_corpus_uri:
                raise ValueError("RAG_CORPUS_URI is required for research service")
            retriever = GCSGroundedRetriever(
                corpus_uri=settings.rag_corpus_uri,
                project=settings.google_cloud_project,
                expected_sha256=settings.rag_corpus_sha256,
            )
    return A2AOperationService(
        role=settings.service_role,
        research_generator=research_generator,
        evidence_loader=evidence_loader,
        retriever=retriever,
        objective_executor=objective_executor,
    )


def create_a2a_app(
    *,
    settings: Settings | None = None,
    research_generator: StructuredResearchGenerator | None = None,
    evidence_loader: ResearchEvidenceLoader | None = None,
    retriever: GroundedRetriever | None = None,
    objective_executor: ObjectiveEvidenceExecutor | None = None,
) -> Starlette:
    """Expose a typed operation service through the official ADK A2A adapter."""

    try:
        from google.adk.a2a.utils.agent_to_a2a import to_a2a
    except ImportError as exc:  # pragma: no cover - cloud extra only
        raise RuntimeError("install the cloud dependency extra to serve A2A") from exc

    runtime = settings or get_settings()
    operation_service = _build_operation_service(
        runtime,
        research_generator=research_generator,
        evidence_loader=evidence_loader,
        retriever=retriever,
        objective_executor=objective_executor,
    )
    port = int(os.getenv("PORT", "8080"))
    agent = build_service_agent(runtime)
    kwargs: dict[str, Any] = {
        "port": port,
        "agent_executor_factory": lambda _runner: build_operation_agent_executor(operation_service),
    }
    if runtime.public_service_url:
        from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

        public_url = runtime.public_service_url.rstrip("/") + "/"
        parsed = urlsplit(public_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("PUBLIC_SERVICE_URL must be an absolute HTTPS URL")
        card = AgentCard(
            name=agent.name,
            description=agent.description,
            supported_interfaces=[
                AgentInterface(
                    url=public_url,
                    protocol_binding="JSONRPC",
                    protocol_version="1.0",
                )
            ],
            version="0.1.0",
            capabilities=AgentCapabilities(streaming=False),
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
            skills=[
                AgentSkill(
                    id=agent.name,
                    name=agent.name,
                    description=agent.description,
                    tags=["post-training", runtime.service_role, "typed-data"],
                )
            ],
        )
        kwargs.update(
            {
                "host": parsed.hostname,
                "protocol": "https",
                "agent_card": card,
            }
        )
    return to_a2a(agent, **kwargs)


def _bootstrap_a2a_app() -> Starlette:
    """Build the module-level ASGI app and fail fast on incomplete cloud wiring."""

    runtime = get_settings()
    if runtime.service_role == "coordinator":
        # The coordinator imports ``create_a2a_app`` through app.main but never
        # serves this module-level object. Keeping imports side-effect free makes
        # local tests and API startup independent of the optional cloud extra.
        return Starlette()
    return create_a2a_app(settings=runtime)


a2a_app = _bootstrap_a2a_app()
