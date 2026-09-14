# Vietnamese "No Parking" Sign Reference Images (P.131a/b/c)

Collected 2026-09-14. **8 images, 3 classes** — see honest limitations below before using
these for training.

```
normal_131a/   4 images   plain "cấm đỗ xe" (no vertical bars)
odd_131b/      2 images   "cấm đỗ xe ngày lẻ" (1 white bar = odd days)
even_131c/     2 images   "cấm đỗ xe ngày chẵn" (2 white bars = even days)
```

## Sources (all downloaded directly, no scraping tool/bulk crawl — checked one by one)

| File | Source | Type |
|---|---|---|
| `normal_131a/ref_otohathanh_131a.jpg` | otohathanh.com | clean illustration |
| `normal_131a/ref_otohathanh_p130_stopno.png` | otohathanh.com | related sign (P.130, no stop+park) — bonus, not one of the 3 target classes |
| `normal_131a/ref_quynhnga_installed.jpg` | congtyquynhnga.com | **real photo**, sign installed on a real VN street |
| `normal_131a/ref_quynhnga_p131.jpg` | congtyquynhnga.com | technical spec drawing (dimensions overlay) |
| `odd_131b/ref_everest_131b.jpg` | everest.org.vn | clean illustration |
| `odd_131b/ref_otohathanh_131b.jpg` | otohathanh.com | clean illustration |
| `even_131c/ref_everest_131c.jpg` | everest.org.vn | clean illustration |
| `even_131c/ref_otohathanh_131c.jpg` | otohathanh.com | clean illustration |

One image (`congtyquynhnga.com/upload/images/cac-bien-bao-cam-do-xe.jpg`, a 3-sign collage) was
**downloaded then deleted** — the source page itself mislabels both the 1-bar and 2-bar signs
as "P.131c," which contradicts the correctly-labeled sources above. Flagging this so nobody
re-adds it from a cache without noticing.

## Honest limitations — read before training anything on this

1. **8 images is nowhere near enough to train a detector.** This is a starter reference set to
   confirm the visual design and prove the folder/pipeline, not a training-ready dataset.
2. **Most of these are clean vector illustrations or technical drawings, not real camera
   photos.** Only `ref_quynhnga_installed.jpg` shows a sign in a real street scene. A detector
   trained only on clean icons will likely fail on real, weathered, angled, security-camera
   footage — exactly the gap between "looks right in a blog post" and "works on your pilot
   video."
3. **The real fix is the Roboflow dataset** (`traffic_signs-vietnam`, 808 images,
   https://universe.roboflow.com/traffic-signs-zk4wy/traffic_signs-vietnam) found earlier —
   but its bulk-download API is behind Cloudflare's bot check and requires a free Roboflow
   account + API key to export. I couldn't create that account on your behalf (needs a real
   email/login). **This is a 2-minute prep task for Hieu**, not a coding task: sign up free at
   roboflow.com, open the dataset, generate an API key, and the actual bulk download becomes a
   one-line script once that key exists.
4. **Still unverified:** whether that 808-image Roboflow dataset separately labels P.131b
   (odd) and P.131c (even) at all, or lumps them into one generic "cấm đỗ xe" class. Check this
   first thing after getting API access — it changes whether the odd/even distinction is
   trainable from that dataset alone or needs supplementing.
