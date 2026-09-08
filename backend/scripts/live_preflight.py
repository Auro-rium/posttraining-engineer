"""Read-only readiness report for one guarded AWS post-training run.

Exit codes: 0 means every check passed; 2 means configuration or provider
readiness is blocked.  This command never writes S3/DynamoDB and never submits
a SageMaker job.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.live_execution import (  # noqa: E402
    LiveExecutionBlocked,
    PreflightRunner,
    config_from_environment,
    safe_json_print,
)


def main() -> int:
    try:
        config = config_from_environment()
        report = PreflightRunner(config).run()
    except LiveExecutionBlocked as exc:
        safe_json_print({"status": "BLOCKED", "reason": str(exc)})
        return 2
    except Exception:
        safe_json_print({"status": "BLOCKED", "reason": "preflight unavailable"})
        return 2
    safe_json_print(report.safe_dict())
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
