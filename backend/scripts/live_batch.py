"""Run up to five guarded AWS candidates, requiring approval for every run."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.live_execution import (  # noqa: E402
    LiveExecutionBlocked,
    LiveExecutionFailed,
    PreflightRunner,
    config_from_environment,
    create_aws_controller,
    safe_json_print,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, choices=range(1, 6))
    parser.add_argument(
        "--approval-token",
        action="append",
        default=[],
        help="one token per run; omit to enter each token privately",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = config_from_environment()
        if args.runs > config.max_runs:
            raise LiveExecutionBlocked("requested batch exceeds configured run cap")
        report = PreflightRunner(config).run()
        if not report.ready:
            safe_json_print({"status": "BLOCKED", "reason": "preflight is BLOCKED"})
            return 2
        controller = create_aws_controller(config)
        summaries: list[dict[str, object]] = []
        for run_number in range(1, args.runs + 1):
            if len(args.approval_token) >= run_number:
                token = args.approval_token[run_number - 1]
            elif args.approval_token:
                raise LiveExecutionBlocked("one approval token is required for each run")
            else:
                token = getpass.getpass(f"Approval token for run {run_number}: ")
            summary = controller.run_once(run_number=run_number, approval_token=token)
            summaries.append(summary.safe_dict())
    except LiveExecutionBlocked as exc:
        safe_json_print({"status": "BLOCKED", "reason": str(exc)})
        return 2
    except LiveExecutionFailed:
        safe_json_print({"status": "FAILED", "reason": "live provider execution failed"})
        return 1
    except (EOFError, KeyboardInterrupt):
        safe_json_print({"status": "CANCELLED", "reason": "approval input cancelled"})
        return 130
    except Exception:
        safe_json_print({"status": "FAILED", "reason": "live batch unavailable"})
        return 1
    safe_json_print({"status": "COMPLETED", "runs": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
