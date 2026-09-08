"""Execute exactly one explicitly approved, real AWS candidate run."""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument("--run-number", type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        "--approval-token",
        default=os.getenv("LIVE_APPROVAL_TOKEN_INPUT", ""),
        help="per-run token; never printed or persisted by this script",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = config_from_environment()
        report = PreflightRunner(config).run()
        if not report.ready:
            safe_json_print({"status": "BLOCKED", "reason": "preflight is BLOCKED"})
            return 2
        controller = create_aws_controller(config)
        if not args.approval_token:
            packet = controller.build_approval_packet(run_number=args.run_number)
            safe_json_print(
                {
                    "status": "WAITING_APPROVAL",
                    "approval_packet": packet.model_dump(mode="json"),
                    "packet_sha256": packet.digest,
                    "instruction": (
                        "Sign this packet with the configured approval service, then rerun "
                        "with --approval-token"
                    ),
                }
            )
            return 2
        summary = controller.run_once(
            run_number=args.run_number,
            approval_token=args.approval_token,
        )
    except LiveExecutionBlocked as exc:
        safe_json_print({"status": "BLOCKED", "reason": str(exc)})
        return 2
    except LiveExecutionFailed:
        safe_json_print({"status": "FAILED", "reason": "live provider execution failed"})
        return 1
    except Exception:
        safe_json_print({"status": "FAILED", "reason": "live run unavailable"})
        return 1
    safe_json_print(summary.safe_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
