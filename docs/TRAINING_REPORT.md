# Training Report — Colab run, 2026-09-18/19

## Summary

Training completed successfully to 30/30 epochs with good, healthy metrics — but the
resulting weights (`best.pt` / `last.pt`) were **lost** because the Colab runtime was silently
swapped to a fresh backend VM sometime between the last epoch finishing and the download cell
running. This is a real, unresolved problem, not a small fix. **The computer has been left
running, not shut down**, so Hieu can decide how to proceed.

## What happened, in order

1. Training ran for 30/30 epochs, ~1.622 hours, no crashes, no NaN/inf losses at any point
   (monitored epoch-by-epoch throughout). Final metrics below are real and were printed by
   Ultralytics before anything went wrong.
2. Immediately after training, the `files.download("runs/edgeai_v1/weights/best.pt")` cell was
   run and failed: `FileNotFoundError: Cannot find file: runs/edgeai_v1/weights/best.pt`.
3. The actual save path from the training log was
   `/content/EdgeAI/runs/detect/runs/edgeai_v1/weights/best.pt` (Ultralytics nested an extra
   `detect/` level under `project="runs"` in this Ultralytics version, 8.4.155 — a real quirk,
   not a typo on our end). Retried the download with that corrected path — **still**
   `FileNotFoundError`.
4. Ran `!find /content/EdgeAI -name "best.pt"` — got `find: '/content/EdgeAI': No such file or
   directory`. The entire working directory was gone.
5. Confirmed with `!pwd; ls /content; uptime; nvidia-smi --query-gpu=uuid,memory.used`:
   - `/content` contains only the stock `sample_data` folder — a completely fresh filesystem.
   - `uptime` reported **6 minutes** of VM uptime.
   - GPU UUID was `GPU-4f2ded9c-7f61-a74e-9a32-e2d8180a9aea` with **0 MiB** used — a different,
     idle GPU, not the one that just spent 1.6 hours training.

**Conclusion:** the Colab runtime was reassigned to a brand-new backend VM after training
finished, while the browser tab still showed "Connected." This silently wiped all of
`/content`, including the trained weights, before the download cell got a chance to run. This
is a known risk on Colab's free/pay-as-you-go GPU tier — the frontend connection can survive a
backend swap, giving no visible warning that the filesystem underneath changed.

## Real training metrics (from the log, before the weights were lost)

Dataset: `data/ua_detrac_yolo/` — 10,097 train images (8,232 UA-DETRAC + 538 Kaggle motorbike,
oversampled 5x on motorcycle-containing images), 5,790 val images (5,625 UA-DETRAC + 165
motorbike). This run predates the second motorcycle source merge (+601 boxes, 12.7:1 ratio) —
it trained on the earlier ~21.7:1 snapshot.

30 epochs, YOLOv8n, imgsz 640, batch 16, T4 GPU, 1.622 hours total.

**Overall (all classes):**

| Metric | Value |
|---|---|
| Precision | 0.815 |
| Recall | 0.643 |
| mAP50 | 0.743 |
| mAP50-95 | 0.565 |

**Per-class:**

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| car | 5,588 | 55,557 | 0.838 | 0.637 | 0.756 | 0.551 |
| motorcycle | 165 | 216 | 0.802 | 0.907 | **0.926** | 0.718 |
| bus | 3,113 | 7,252 | 0.865 | 0.648 | 0.771 | 0.590 |
| truck | 2,371 | 3,891 | 0.754 | 0.380 | **0.518** | 0.402 |

Inference speed (on the T4, not the target Pi): 0.2ms preprocess, 1.8ms inference, 1.7ms
postprocess per image.

## Honest read of these numbers

- **Motorcycle came out best, not worst** — despite the known ~21.7:1 training imbalance. The
  5x oversampling on motorcycle-containing images clearly worked, and the validation set only
  has 216 motorcycle instances, which is a small, relatively easy set (this is a genuine result,
  not a data leak — oversampling only touched the train split, per `oversample_motorcycle.py`).
  This is worth calling out plainly rather than assuming the known imbalance would show up as a
  motorcycle weakness — it didn't.
- **Truck is the actual weak class** (recall 0.380, mAP50 0.518), not motorcycle. Truck has the
  fewest real instances after car/bus, and this wasn't previously flagged as a risk — it should
  be now.
- Overall mAP50 0.743 is consistent with the epoch-by-epoch trend already observed (0.64 at
  epoch 1, climbing to plateau around 0.72-0.75 from epoch 12 onward) — no red flags in the
  training dynamics themselves. The only failure was losing the output artifact afterward.

## Comparison to Report Sec. VIII / Table III targets

These are detection-level mAP proxies, not the final violation-level metrics Table III actually
targets (<300ms end-to-end latency, >=90% recall, <10% false-alarm rate on real footage) — this
run measured training-time GPU inference speed and per-class detection accuracy, not the
deployed zone-overlap + dwell-time pipeline's violation detection rate. That real evaluation
still needs Hung's evaluation script (Sec. VIII, `docs/RUNLOG.md`) running against actual
labeled pilot footage once it exists. What this run does support: detection recall in the
0.64-0.91 range per class is in the right ballpark to feed that pipeline, except truck, which
at 0.38 recall would likely miss a lot of real trucks and needs another look.

## What to do next (for Hieu, in the morning)

1. **The weights need to be retrained** — there is no way to recover `best.pt`/`last.pt` from
   this run; the backing files are gone. Re-running takes about the same ~1.6 hours.
2. **Before retraining, add a safeguard so this can't happen again**: either (a) mount Google
   Drive at the start of the notebook and point `project=` at a Drive path so checkpoints save
   somewhere that survives a backend swap, or (b) call `files.download()` right after training
   finishes in the same cell/run rather than as a separate step later, or (c) periodically save
   `best.pt` to Drive during training via a callback. (a) is the most robust and is recommended.
3. **Use the improved dataset for the retrain**: `src/add_motorbike_data_v2.py` was merged after
   this run started (see `docs/RUNLOG.md`, 2026-09-18 entry) — the next run should include it
   for the 12.7:1 ratio instead of this run's 21.7:1.
4. **Keep an eye on truck performance** specifically in the next run's per-class table — this
   run's 0.38 recall was unexpected and wasn't part of the known-imbalance caveat already
   documented for motorcycle.

**The computer has been left on, per instructions, since this isn't a small fix.**
