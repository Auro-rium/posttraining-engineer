"""Credential-gated Google Cloud integrations.

Imports are intentionally lazy so local demo mode and unit tests never require
Google credentials or the heavyweight cloud dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.settings import Settings


class CloudIntegrationError(RuntimeError):
    """Raised when a verifiable cloud operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class VertexJobHandle:
    """Stable evidence returned after a Vertex job has been submitted."""

    resource_name: str
    display_name: str
    state: str
    console_uri: str | None = None


class SecretManagerReader:
    """Read a named secret version without exposing it to logs or state."""

    def __init__(self, settings: Settings) -> None:
        if not settings.google_cloud_project:
            raise CloudIntegrationError("GOOGLE_CLOUD_PROJECT is required")
        self._project = settings.google_cloud_project

    def access(self, secret_id: str, version: str = "latest") -> str:
        try:
            from google.cloud import secretmanager
        except ImportError as exc:  # pragma: no cover - depends on cloud extra
            raise CloudIntegrationError("install the cloud dependency extra") from exc

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{self._project}/secrets/{secret_id}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8")


class VertexTrainingLauncher:
    """Submit bounded QLoRA containers to Vertex AI Custom Training."""

    def __init__(self, settings: Settings) -> None:
        if settings.environment != "cloud":
            raise CloudIntegrationError("Vertex jobs are only available in cloud mode")
        self._settings = settings

    def submit(
        self,
        *,
        run_id: str,
        experiment_id: str,
        dataset_uri: str,
        output_uri: str,
        qlora_args: dict[str, int | float | str],
    ) -> VertexJobHandle:
        """Submit one asynchronous L4 job and return its immutable resource name."""

        try:
            from google.cloud import aiplatform
        except ImportError as exc:  # pragma: no cover - depends on cloud extra
            raise CloudIntegrationError("install the cloud dependency extra") from exc

        settings = self._settings
        if not all(
            [
                settings.google_cloud_project,
                settings.vertex_staging_bucket,
                settings.training_container_uri,
            ]
        ):
            raise CloudIntegrationError("Vertex training settings are incomplete")
        container_uri = settings.training_container_uri
        if container_uri is None:
            raise CloudIntegrationError("TRAINING_CONTAINER_URI is required")

        aiplatform.init(
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            staging_bucket=settings.vertex_staging_bucket,
        )
        display_name = f"apte-{run_id[-8:]}-{experiment_id[-8:]}"
        job = aiplatform.CustomContainerTrainingJob(
            display_name=display_name,
            container_uri=container_uri,
            staging_bucket=settings.vertex_staging_bucket,
        )
        args: list[str | float | int] = [
            "--run-id",
            run_id,
            "--experiment-id",
            experiment_id,
            "--dataset-uri",
            dataset_uri,
            "--output-uri",
            output_uri,
        ]
        if settings.hf_secret_id:
            args.extend(["--hf-secret-id", settings.hf_secret_id])
        for key, value in sorted(qlora_args.items()):
            args.extend([f"--{key.replace('_', '-')}", str(value)])

        job.run(
            args=args,
            replica_count=1,
            machine_type="g2-standard-4",
            accelerator_type="NVIDIA_L4",
            accelerator_count=1,
            sync=False,
            enable_web_access=False,
        )
        resource_name = getattr(job, "resource_name", None)
        if not resource_name:
            raise CloudIntegrationError("Vertex did not return a job resource name")
        return VertexJobHandle(
            resource_name=str(resource_name),
            display_name=display_name,
            state="SUBMITTED",
            console_uri=(
                "https://console.cloud.google.com/vertex-ai/training/custom-jobs/"
                f"{str(resource_name).rsplit('/', 1)[-1]}?project={settings.google_cloud_project}"
            ),
        )

    def state(self, resource_name: str) -> str:
        """Return the current Vertex state without mutating the job."""

        try:
            from google.cloud import aiplatform
        except ImportError as exc:  # pragma: no cover
            raise CloudIntegrationError("install the cloud dependency extra") from exc

        job = aiplatform.CustomJob.get(resource_name)
        return str(job.state.name if hasattr(job.state, "name") else job.state)


def safe_cloud_metadata(value: Any) -> str | int | float | bool | None:
    """Allow only scalar cloud metadata into persistent research state."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
