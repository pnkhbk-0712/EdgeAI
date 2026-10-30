"""Merge PPE sources into data/ppe_merged/ with one class scheme (2026-10-24).

Classes: person, helmet, no_helmet, vest, no_vest, gloves, no_gloves, goggles, no_goggles.
Boots/no_boots are dropped (out of scope); an image left with no box is dropped.

Sources (ledger with licences: docs/DATASET_AUDIT_PPE.md):
  cppe  - Ultralytics Construction-PPE (AGPL-3.0). Real worksite photos; its own val/test are kept as
          the evaluation sets, minus images with a near-twin (dHash <= TWIN_BITS) anywhere in train.
          Class `none` is mapped to no_vest: crops show torsos without a hi-vis vest (inferred from
          sample crops, not stated in the docs -- flagged in the audit).
  ppev5 - Roboflow PPEv5 v16 (CC BY 4.0). Close-up body-part photos; ALL of it goes to train (its
          own valid has a different class mix from its train, so it is no use as validation).

Outputs: <out>/{train,valid,test}/{images,labels} (images hard-linked), train.txt (each image once),
train_repeat.txt (rare classes repeated, LVIS-style repeat factor), train_cppe.txt (ablation A:
Construction-PPE only), data.yaml / data_repeat.yaml / data_cppe.yaml.
`path:` in the yamls is absolute (Ultralytics resolves a relative path against its own datasets dir, not the
yaml); on Colab rewrite it -- see docs/PPE_TRAINING.md.
Usage: python src/build_ppe_dataset.py
"""
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from imghash import dhash, hamming

SRC = Path("data/ppe_sources")
OUT = Path("data/ppe_merged")
NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "gloves", "no_gloves", "goggles", "no_goggles"]
TWIN_BITS = 8          # same threshold as the helmet audit: <= 8 of 64 bits counts as a twin
REPEAT_T, REPEAT_MAX = 0.15, 4

SOURCES = [
    dict(name="cppe", root=SRC / "construction_ppe" / "raw", layout="ultra",
         splits={"train": "train", "val": "valid", "test": "test"},
         map={"helmet": "helmet", "gloves": "gloves", "vest": "vest", "goggles": "goggles", "none": "no_vest",
              "Person": "person", "no_helmet": "no_helmet", "no_goggle": "no_goggles", "no_gloves": "no_gloves"}),
    dict(name="ppev5", root=SRC / "ppev5" / "v16", layout="roboflow",
         splits={"train": "train", "valid": "train"},
         map={"glasses": "goggles", "gloves": "gloves", "helmet": "helmet", "no glasses": "no_goggles",
              "no gloves": "no_gloves", "no helmet": "no_helmet", "no vest": "no_vest", "person": "person",
              "vest": "vest"}),
    # SH17 (CC BY-NC-SA 4.0): only the gloves/glasses images, shrunk by src/prep_sh17.py. Train only, and kept out of
    # train.txt / train_sampled.txt so configs A and C' stay exactly as they were; used by train_sh17.txt (C'').
    dict(name="sh17", root=SRC / "sh17" / "raw", layout="ultra", splits={"all": "train"},
         map={"person": "person", "glasses": "goggles", "gloves": "gloves", "helmet": "helmet", "safety-vest": "vest"}),
]


# SH17 images where helmet/vest is visible but unlabeled (src/scan_sh17_missing_labels.py) would teach "no vest/helmet"
_ex = SRC / "sh17" / "exclude_unlabeled.txt"
SH17_EXCLUDE = set(_ex.read_text().split()) if _ex.exists() else set()


def src_names(root):
    n = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)


def pairs(root, layout, split):
    img_dir = root / "images" / split if layout == "ultra" else root / split / "images"
    lbl_dir = root / "labels" / split if layout == "ultra" else root / split / "labels"
    for img in sorted(img_dir.iterdir()):
        lf = lbl_dir / (img.stem + ".txt")
        if img.suffix.lower() in (".jpg", ".jpeg", ".png") and lf.exists():
            yield img, lf


def convert(lf, id_map):
    out = []
    for ln in lf.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5 and int(p[0]) in id_map:
            out.append(f"{id_map[int(p[0])]} {' '.join(p[1:5])}")
    return out


# 1. collect candidates per target split
items = {"train": [], "valid": [], "test": []}   # (source, img, converted label lines)
for s in SOURCES:
    names = src_names(s["root"])
    id_map = {i: NAMES.index(s["map"][n]) for i, n in enumerate(names) if n in s["map"]}
    dropped = [n for n in names if n not in s["map"]]
    print(f"{s['name']}: mapped {len(id_map)} classes, dropped {dropped}")
    for src_split, dst in s["splits"].items():
        for img, lf in pairs(s["root"], s["layout"], src_split):
            lines = convert(lf, id_map)
            if s["name"] == "sh17" and img.stem in SH17_EXCLUDE:
                continue
            if lines:
                items[dst].append((s["name"], img, lines))

# 2. evaluation sets: drop images that have a near-twin anywhere in train (SH17 excluded here so the eval sets stay
# identical to the earlier runs; instead any SH17 image that is a twin of a kept eval image is dropped from train)
sh17_items = [i for i in items["train"] if i[0] == "sh17"]
items["train"] = [i for i in items["train"] if i[0] != "sh17"]
train_h = np.stack([dhash(i[1]) for i in items["train"]])
for split in ("valid", "test"):
    keep = []
    for it in items[split]:
        if hamming(dhash(it[1]), train_h).min() > TWIN_BITS:
            keep.append(it)
    print(f"{split}: {len(items[split])} images, {len(items[split]) - len(keep)} dropped as near-twins of train")
    items[split] = keep
eval_h = np.stack([dhash(i[1]) for sp in ("valid", "test") for i in items[sp]])
sh17_keep = [i for i in sh17_items if hamming(dhash(i[1]), eval_h).min() > TWIN_BITS]
print(f"sh17: {len(sh17_items)} images, {len(sh17_items) - len(sh17_keep)} dropped as near-twins of valid/test")
items["train"] += sh17_keep

# 3. write
if OUT.exists():
    shutil.rmtree(OUT)
for split, lst in items.items():
    (OUT / split / "images").mkdir(parents=True)
    (OUT / split / "labels").mkdir(parents=True)
    for srcname, img, lines in lst:
        stem = f"{srcname}_{img.stem}".replace("(", "").replace(")", "").replace(" ", "_")
        dst = OUT / split / "images" / (stem + img.suffix.lower())
        try:
            os.link(img, dst)
        except OSError:
            shutil.copy2(img, dst)
        (OUT / split / "labels" / (stem + ".txt")).write_text("\n".join(lines) + "\n")

# 4. train lists: plain and repeat-factor
all_train = sorted((OUT / "train" / "images").iterdir())
sh17_files = [f for f in all_train if f.name.startswith("sh17_")]
train_files = [f for f in all_train if not f.name.startswith("sh17_")]
cls_of = {f: {int(l.split()[0]) for l in (OUT / "train" / "labels" / (f.stem + ".txt")).read_text().split("\n") if l} for f in train_files}
freq = Counter(c for s in cls_of.values() for c in s)
rep_c = {c: max(1.0, math.sqrt(REPEAT_T / (freq[c] / len(train_files)))) for c in freq}
rel = lambda f: "./" + f.relative_to(OUT).as_posix()
(OUT / "train.txt").write_text("\n".join(rel(f) for f in train_files) + "\n")
reps = {f: min(REPEAT_MAX, round(max(rep_c[c] for c in cs))) for f, cs in cls_of.items()}
(OUT / "train_repeat.txt").write_text("\n".join(rel(f) for f in train_files for _ in range(reps[f])) + "\n")
print("repeat factors per class:", {NAMES[c]: round(r, 2) for c, r in sorted(rep_c.items())})
print(f"train images {len(train_files)} -> {sum(reps.values())} lines with repeats")

# ablation A: real-worksite images only (Construction-PPE), no close-up supplement
cppe_files = [f for f in train_files if f.name.startswith("cppe_")]
(OUT / "train_cppe.txt").write_text("\n".join(rel(f) for f in cppe_files) + "\n")
print(f"train_cppe.txt: {len(cppe_files)} images (ablation A)")

# configuration C': all of Construction-PPE + a class-balanced sample of PPEv5 (weighted sampling without
# replacement, weight = 1/frequency of the image's rarest class) -- C is ~6x slower to train than A.
PPEV5_SAMPLE = 2500
rng = np.random.default_rng(0)
v5 = [f for f in train_files if f.name.startswith("ppev5_")]
w = np.array([1.0 / freq[min(cls_of[f], key=lambda c: freq[c])] for f in v5])
keys = rng.random(len(v5)) ** (1.0 / w)
picked = [v5[i] for i in np.argsort(-keys)[:PPEV5_SAMPLE]]
sampled = sorted(cppe_files + picked)
(OUT / "train_sampled.txt").write_text(chr(10).join(rel(f) for f in sampled) + chr(10))
box = Counter(int(l.split()[0]) for f in sampled for l in (OUT / "train" / "labels" / (f.stem + ".txt")).read_text().split(chr(10)) if l)
print(f"train_sampled.txt: {len(sampled)} images ({len(cppe_files)} cppe + {len(picked)} ppev5); boxes:",
      {NAMES[c]: n for c, n in sorted(box.items())})

# configuration C'': C' plus all selected SH17 gloves/glasses images (more gloves/goggles boxes)
(OUT / "train_sh17.txt").write_text(chr(10).join(rel(f) for f in sampled + sh17_files) + chr(10))
box2 = Counter(int(l.split()[0]) for f in sampled + sh17_files for l in (OUT / "train" / "labels" / (f.stem + ".txt")).read_text().split(chr(10)) if l)
print(f"train_sh17.txt: {len(sampled) + len(sh17_files)} images ({len(sh17_files)} sh17); boxes:", {NAMES[c]: n for c, n in sorted(box2.items())})

for fname, tr in (("data_sh17.yaml", "train_sh17.txt"), ("data.yaml", "train.txt"), ("data_repeat.yaml", "train_repeat.txt"), ("data_cppe.yaml", "train_cppe.txt"), ("data_sampled.yaml", "train_sampled.txt")):
    (OUT / fname).write_text(yaml.safe_dump(
        {"path": OUT.resolve().as_posix(), "train": tr, "val": "valid/images", "test": "test/images", "names": dict(enumerate(NAMES))},
        sort_keys=False), encoding="utf-8")
print("wrote", OUT)
