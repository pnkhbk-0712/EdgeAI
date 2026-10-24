"""Correction/appeal tool for data/samples/helmet_events.jsonl (Report Step 8, Ethics).

No identity is attached to a track_id (by design -- see demo_helmet.py PRIVACY_NOTE), so a
worker can't appeal "my event" by name. The real-world flow this supports: a site safety officer
reviewing the on-screen/log track_id + timestamp of a flagged violation (e.g. a false "no helmet"
from a low-confidence or occluded frame) marks it reviewed and records the correction -- the
original event is kept, not edited away, so the log stays an honest audit trail.

Usage:
    python src/review_events.py --track-id 7 --timestamp 02:13 --item vest --correction false_positive --note "vest hidden by crate, not missing"
    python src/review_events.py --list-unreviewed
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

EVENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "helmet_events.jsonl"
VIOLATION_EVENTS = ("ppe_violation", "no_helmet_violation")  # the latter = pre-2026-10-24 helmet-only logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-id", type=int)
    parser.add_argument("--timestamp", help="mm:ss as printed in the event, e.g. 02:13")
    parser.add_argument("--item", help="missing PPE item (helmet|vest|gloves|goggles) -- needed when one person "
                                       "has several events at the same timestamp")
    parser.add_argument("--correction", choices=["confirmed", "false_positive"])
    parser.add_argument("--note", default="")
    parser.add_argument("--list-unreviewed", action="store_true")
    args = parser.parse_args()

    if not EVENTS_PATH.exists():
        raise SystemExit(f"No events log at {EVENTS_PATH} yet -- run demo_helmet.py first.")
    events = [json.loads(line) for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.list_unreviewed:
        pending = [e for e in events if e.get("event") in VIOLATION_EVENTS and not e.get("reviewed")]
        if not pending:
            print("No unreviewed violation events.")
        for e in pending:
            print(f"track_id={e['track_id']} item={e.get('missing_item', 'helmet')} zone={e['zone']} timestamp={e['timestamp']} "
                  f"confidence={e['confidence']} band={e['confidence_band']}")
        return

    if args.track_id is None or args.timestamp is None or args.correction is None:
        parser.error("--track-id, --timestamp and --correction are required unless --list-unreviewed")

    matches = [e for e in events if e.get("track_id") == args.track_id and e.get("timestamp") == args.timestamp
               and e.get("event") in VIOLATION_EVENTS
               and (args.item is None or e.get("missing_item", "helmet") == args.item)]
    if not matches:
        raise SystemExit(f"No event found with track_id={args.track_id} timestamp={args.timestamp}")

    for e in matches:
        e["reviewed"] = True
        e["correction"] = args.correction
        e["review_note"] = args.note
        e["reviewed_at"] = datetime.now().isoformat(timespec="seconds")

    EVENTS_PATH.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    print(f"Marked {len(matches)} event(s) reviewed: correction={args.correction}")


if __name__ == "__main__":
    main()
