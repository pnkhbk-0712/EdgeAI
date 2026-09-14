# Roboflow `traffic_signs-vietnam` — real findings (2026-09-14)

Downloaded with Hieu's API key (not committed anywhere — used only as an env var for the
download, then discarded). **This folder is git-ignored** (41MB, 808 images across 54 classes)
— re-download with the script below if needed on another machine.

## The answer to the open question: odd/even day signs are NOT separately labeled

54 total classes (`Cam do xe`, `Cam dung va do xe`, plus 52 other unrelated prohibition/warning
signs — speed limits, no U-turn, no motorbikes, etc.). **There is exactly one class for "no
parking," `Cam do xe`, with 56 labeled images — and every one checked is the plain P.131a
variant (no vertical bars).** Manually cropped and visually inspected 3 of the 56 to confirm;
none showed the 1-bar (odd/P.131b) or 2-bar (even/P.131c) design.

**Conclusion:** this dataset can train a detector for "is there a no-parking sign here at all,"
but **cannot** by itself teach a model to distinguish odd-day vs. even-day vs. unconditional
no-parking signs — those three visually distinct classes just aren't represented as separate
labels here.

## What this means for the odd/even feature

To actually classify P.131a vs. P.131b vs. P.131c, the team needs one of:
1. **Manually re-label** a subset of these 56 (or the starter set in `../signs/`) into the 3
   sub-classes by hand — since the bar count is easy for a human to see, this is realistic
   for a small set (tens of images) in an afternoon, not a big ask.
2. **Collect a few more real examples** of the 1-bar and 2-bar variants specifically (they're
   rarer in the wild than plain no-parking signs) — a pilot-site filming trip is a natural
   place to grab a couple of real photos if any exist near the target site.
3. Fall back to the plan already in the report/artifact: detect "no-parking sign present," and
   let a human still configure the odd/even schedule by hand for that specific zone (the
   Design Note's `active_days` field) — this remains the safe default if 1-2 don't pan out in
   time.

## Re-download script (needs a Roboflow API key — get one free at roboflow.com)

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY_HERE")  # never commit a real key
project = rf.workspace("traffic-signs-zk4wy").project("traffic_signs-vietnam")
project.version(1).download("yolov8", location="signs_roboflow")
```

## Dataset split sizes

| Split | Images |
|---|---|
| train | 566 |
| valid | 156 |
| test | 86 |

License: CC BY 4.0 (per `data.yaml`).
