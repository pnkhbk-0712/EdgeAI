# Events log — retention, correction/appeal, access governance (Report Step 8)

Closes the 3 named gaps in Step 8 (`Project_Report_EdgeAI_Group9.docx`, mục 11 Ưu tiên 2):
retention policy, appeal/correction mechanism, governance on who can access the log. All three
below are real, implemented or decided — not aspirational.

## 1. Retention policy — implemented

`src/demo_helmet.py`, `EVENT_RETENTION_DAYS = 30`. On every run, `load_and_purge_events()` drops
any event older than 30 days before appending new ones (2026-10-13, see `docs/RUNLOG.md`). 30
days is a starting default for a classroom prototype, not matched to any real regulatory
requirement — change the one constant if the team adopts a different rule later.

## 2. Correction/appeal mechanism — implemented

No identity is attached to a `track_id` (by design, for privacy — see `demo_helmet.py`
`PRIVACY_NOTE`), so there is no per-worker "my event" to appeal by name. The real-world flow this
supports instead: the on-screen label and the log both print `track_id` + `timestamp` for every
flagged violation; a site safety officer who judges a specific flag wrong (e.g. low-confidence
`[UNCERTAIN - review]` frame, occlusion) reviews it and records the correction with:

```
python src/review_events.py --track-id 7 --timestamp 02:13 --correction false_positive --note "hat off-frame, not missing"
python src/review_events.py --list-unreviewed   # see what's still pending review
```

The original event is never edited or deleted — `reviewed`, `correction`, `review_note`,
`reviewed_at` are added alongside it, so the log stays a complete audit trail (what the model
flagged AND what a human later decided), not a record that can be silently altered.

## 3. Access governance — decided

For this course prototype: the raw `data/samples/helmet_events.jsonl` (per-event detail —
zone, track_id, confidence, timestamp) is visible only to the project team and the site safety
officer role played in the demo. It is not shared publicly or submitted as part of the report
attachments. What **is** shareable more broadly (report, demo, any future stakeholder update) is
the periodic `compliance_snapshot` events already logged every `COMPLIANCE_SNAPSHOT_INTERVAL_SEC`
(30s) — an aggregate percentage, not per-event detail — matching the "only a
zone/timestamp/confidence event is logged, never raw video" privacy posture already stated in
Step 8 of the report.

## Tài liệu liên quan

- `src/demo_helmet.py` — retention + event schema (`reviewed` field existed but nothing set it
  before this).
- `src/review_events.py` — correction/appeal CLI, new 2026-10-18.
- `docs/RUNLOG.md` 2026-10-18 — full log entry.
- `Project_Report_EdgeAI_Group9.docx`, Step 8 / mục 11 — report section this closes; Hạnh updates
  the Step 8 score and mục 11 status once reviewed.
