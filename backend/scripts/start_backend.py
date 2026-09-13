"""Start the backend, staging the immutable model before objective service startup."""

from __future__ import annotations

from os import environ, execvp
from typing import NoReturn

from scripts.bootstrap_objective_checkpoint import bootstrap_checkpoint


def main() -> NoReturn:
    """Bootstrap the objective model if needed, then replace this process with Uvicorn."""

    if environ.get("SERVICE_ROLE", "coordinator").strip().lower() == "objective":
        # A failure is intentionally fatal: never expose a healthy-looking
        # objective service without the exact, verified model snapshot.
        bootstrap_checkpoint()

    port = environ.get("PORT", "8080")
    execvp(
        "uvicorn",
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", port],
    )


if __name__ == "__main__":
    main()
