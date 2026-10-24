"""Figures for docs/BaoCao_Tuan11_Step1_Step2*.docx -> docs/figures/. Numbers are the measured ones in docs/RUNLOG.md
and docs/DATASET_AUDIT_WEEK11.md. Usage: python src/make_report_figures.py
"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

OUT = Path("docs/figures"); OUT.mkdir(exist_ok=True)
BY = Path("data/helmet_roles_by_class")
RB = Path("data/helmet_roboflow")
random.seed(7)

# Fig 1: Pi latency + head AP50 (measured: Pi 4B 21/10/2026; val set 1413 imgs)
cfg = ["fp32\n640", "INT8\n640", "fp32\n416", "fp32\n320"]
ms = [472.1, 553.3, 234.4, 129.7]
ap = [0.960, 0.957, 0.936, 0.896]
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
a.bar(cfg, ms, color=["#c0504d", "#c0504d", "#4f81bd", "#4f81bd"])
a.axhline(300, color="k", ls="--", lw=1); a.text(3.45, 308, "300 ms", ha="right", fontsize=8)
for i, v in enumerate(ms): a.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=8)
a.set_ylabel("ms / khung (Pi 4B, CPU)"); a.set_title("Độ trễ suy luận trên Pi", fontsize=10)
b.bar(cfg, ap, color="#9bbb59"); b.set_ylim(0.8, 1.0); b.axhline(0.8, color="k", ls="--", lw=1)
for i, v in enumerate(ap): b.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
b.set_ylabel("AP50 lớp head (val 1.413 ảnh)"); b.set_title("Độ chính xác lớp head", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_pi_latency_ap.png", dpi=150); plt.close(fig)

# Fig 2: class balance (train boxes)
fig, ax = plt.subplots(figsize=(5.5, 3))
names, vals = ["helmet", "head", "person"], [13919, 4612, 450]
ax.bar(names, vals, color=["#4f81bd", "#c0504d", "#7f7f7f"])
for i, v in enumerate(vals): ax.text(i, v + 200, f"{v:,}".replace(",", "."), ha="center", fontsize=9)
ax.set_ylabel("số đối tượng (train)"); ax.set_title("Phân bố lớp tập train (helmet : person = 31 : 1)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_class_balance.png", dpi=150); plt.close(fig)


def tile(f, s):
    return Image.open(f).convert("RGB").resize((s, s))


def montage(rows, cols, size, label_w=150):
    """rows = [(label, [paths])] -> one image, label column on the left."""
    sh = Image.new("RGB", (label_w + cols * size, len(rows) * size), "white"); d = ImageDraw.Draw(sh)
    for r, (lab, fl) in enumerate(rows):
        d.text((6, r * size + size // 2 - 6), lab, fill="black")
        for c, f in enumerate(fl[:cols]):
            sh.paste(tile(f, size), (label_w + c * size, r * size))
    return sh


# Fig 3: near-duplicates, valid | train (pairs checked by eye; dHash distance 0)
pairs = [("004977_jpg.rf.a10856f96d56ab549c2d36b554bbd19a.jpg", "002844_jpg.rf.35ea397b1561548e5d2ae1d5b30f98b6.jpg"),
         ("006124_jpg.rf.1866aad6305e5086d7925fb70a435b15.jpg", "002187_jpg.rf.2c133467dd80031d2b0211678413d131.jpg"),
         ("005239_jpg.rf.5f1ecfd1a2f522498672fba5e0e0523f.jpg", "000496_jpg.rf.844c8d31c6dd72d46bd2e043282f8322.jpg")]
W = 260
sh = Image.new("RGB", (2 * W + 10, len(pairs) * (W * 3 // 4) + 22), "white"); d = ImageDraw.Draw(sh)
d.text((W // 2 - 20, 4), "valid", fill="black"); d.text((W + 10 + W // 2 - 20, 4), "train", fill="black")
for r, (v, t) in enumerate(pairs):
    y = 20 + r * (W * 3 // 4)
    sh.paste(Image.open(RB / "valid/images" / v).convert("RGB").resize((W, W * 3 // 4 - 2)), (0, y))
    sh.paste(Image.open(RB / "train/images" / t).convert("RGB").resize((W, W * 3 // 4 - 2)), (W + 10, y))
sh.save(OUT / "fig_near_duplicates.png")

# Fig 4: colour folders (random 8 crops per colour, train)
cols = ["white", "yellow", "orange", "red", "blue", "green", "unclear"]
rows = []
for c in cols:
    fl = sorted((BY / f"helmet_{c}" / "train").glob("*.jpg")); rows.append((f"helmet_{c}", random.sample(fl, min(8, len(fl)))))
montage(rows, 8, 84).save(OUT / "fig_color_folders.png")

# Fig 5: head-wear folders
def pick(d, n, only=None):
    fl = sorted(p for p in d.glob("*.jpg") if only is None or only(p)); return random.sample(fl, min(n, len(fl)))
nh = pick(BY / "no_hat" / "train", 8)
hn_orig = pick(BY / "hat_not_safety" / "train", 8, lambda p: not p.name.startswith("ext_"))
hn_ext = pick(BY / "hat_not_safety" / "train", 8, lambda p: p.name.startswith("ext_"))
montage([("no_hat", nh), ("hat_not_safety\n(dataset gốc)", hn_orig), ("hat_not_safety\n(bộ ngoài cap/hat)", hn_ext)], 8, 96, 170).save(OUT / "fig_headwear.png")
print("ok", [p.name for p in sorted(OUT.glob("*.png"))])
