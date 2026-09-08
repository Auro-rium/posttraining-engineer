"""Sign one immutable live-run approval packet with the operator secret."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.live_execution import ApprovalPacket, issue_approval_token  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", help="JSON packet file, or - to read stdin")
    args = parser.parse_args()
    raw = sys.stdin.read() if args.packet == "-" else Path(args.packet).read_text()
    payload = json.loads(raw)
    packet_payload = payload.get("approval_packet", payload)
    packet = ApprovalPacket.model_validate(packet_payload)
    secret = os.getenv("LIVE_APPROVAL_SECRET", "")
    if not secret:
        raise SystemExit("LIVE_APPROVAL_SECRET is required")
    # The token is intentionally written only to stdout for direct piping into
    # the guarded runner; it is never persisted or logged by this script.
    print(issue_approval_token(packet, secret))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
