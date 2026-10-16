# Helmet model export/quantization benchmark (2026-10-10)

CPU-only, same methodology as `docs/IMPLEMENTATION_REPORT.md`'s parking-project comparison --
real per-frame latency, not just file size, measured on `val_batch0_pred.jpg`
(5 warmup + 30 timed runs per format).

| Format | Size (MB) | ms/frame | FPS |
|---|---|---|---|
| PyTorch (.pt) | 6.23 | 59.0 | 16.9 |
| ONNX (fp32) | 12.27 | 29.7 | 33.7 |
| ONNX (INT8, dynamic) | 3.36 | 45.6 | 21.9 |

**ONNX fp32 is faster than PyTorch by 49.7%. INT8 is slower than ONNX fp32 by 53.8%, despite being 72.6% smaller.**

Acceptance check against Step 3/4's target (<300ms/frame on the Pi 4B -- this is a laptop CPU
number, not yet the Pi itself, see Step 5's "testing on target device" gap):
PASSES the latency target on ONNX fp32 at 29.7ms/frame.
