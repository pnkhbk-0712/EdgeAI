"""Convert UA-DETRAC's native XML annotations into a YOLO-trainable dataset.

Real format, confirmed by inspecting the actual downloaded files (not assumed from
memory) -- one XML per sequence:

    <sequence name="MVI_39031">
      <frame num="1">
        <target_list>
          <target id="1">
            <box left="745.6" top="357.33" width="148.2" height="115.14"/>
            <attribute vehicle_type="car" .../>
          </target>
        </target_list>
      </frame>
      ...
    </sequence>

Images live at DETRAC-Images/DETRAC-Images/<sequence>/img#####.jpg (5-digit, 1-indexed,
matching the XML frame's `num`).

Design decisions (so re-runs are predictable, not "whatever it did last time"):

- SAMPLING: converts every Nth frame per sequence (STRIDE, default 10), not all 140,131
  frames. Consecutive video frames are near-duplicates; a laptop fine-tune gets little
  benefit from 10x the near-identical images at 10x the conversion/training time. Raise
  STRIDE to use more data once this subset proves the pipeline works.
- CLASSES: UA-DETRAC only has {car, bus, van, others} -- no motorcycle. Mapped car->car,
  bus->bus, van->truck (closest real-world analog); "others" is dropped by default (too
  ambiguous a bucket to trust as one class) -- see INCLUDE_OTHERS to change that. class id
  1 (motorcycle) is reserved with zero examples for now, so supplementary motorcycle data
  (Report Sec. VI.A) merges into the same class id later without remapping anything.
- Images are COPIED (not symlinked) into a clean images/labels folder pair, since
  Ultralytics' auto label-path derivation (replacing "/images/" in the path) does not
  apply to UA-DETRAC's own "DETRAC-Images/DETRAC-Images/<seq>/" folder names.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copyfile

import cv2

ROOT = Path(__file__).resolve().parent.parent

# Where the dataset actually lives -- see docs/UA_DETRAC_DOWNLOAD.md. Override with the
# env var if you downloaded it somewhere else.
DETRAC_ROOT = Path(os.environ.get(
    "UA_DETRAC_ROOT",
    r"D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\.cache\kagglehub\datasets\bratjay\ua-detrac-orig\versions\2",
))

OUT_ROOT = ROOT / "data" / "ua_detrac_yolo"

SPLITS = {
    "train": ("DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML", "train"),
    "val":   ("DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML",   "val"),
}

STRIDE = 10             # keep 1 in every STRIDE frames per sequence
INCLUDE_OTHERS = False  # True to map UA-DETRAC's "others" class into CLASS_MAP["others"]

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]  # motorcycle stays empty until
                                                       # supplementary data is added
CLASS_MAP = {"car": "car", "bus": "bus", "van": "truck"}
if INCLUDE_OTHERS:
    CLASS_MAP["others"] = "car"  # best-effort fallback, not a confident mapping


def yolo_line(class_id, left, top, w, h, img_w, img_h):
    cx = (left + w / 2) / img_w
    cy = (top + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # clamp -- a few UA-DETRAC boxes run slightly past the frame edge
    cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_split(xml_dir, split_name):
    images_out = OUT_ROOT / "images" / split_name
    labels_out = OUT_ROOT / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    images_root = DETRAC_ROOT / "DETRAC-Images" / "DETRAC-Images"
    xml_files = sorted(xml_dir.glob("*.xml"))

    n_images = 0
    n_boxes = 0
    class_counts = {name: 0 for name in CLASS_NAMES}
    skipped_sequences = []

    for xml_path in xml_files:
        seq_name = xml_path.stem
        seq_img_dir = images_root / seq_name
        if not seq_img_dir.exists():
            skipped_sequences.append(seq_name)
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Read image size once per sequence (fixed camera -> constant resolution),
        # instead of once per frame -- avoids ~14,000 redundant cv2.imread calls.
        first_frame_path = seq_img_dir / "img00001.jpg"
        probe = cv2.imread(str(first_frame_path))
        if probe is None:
            skipped_sequences.append(seq_name)
            continue
        img_h, img_w = probe.shape[:2]

        for frame in root.findall("frame"):
            frame_num = int(frame.get("num"))
            if (frame_num - 1) % STRIDE != 0:
                continue

            src_img = seq_img_dir / f"img{frame_num:05d}.jpg"
            if not src_img.exists():
                continue

            lines = []
            target_list = frame.find("target_list")
            if target_list is not None:
                for target in target_list.findall("target"):
                    box = target.find("box")
                    attr = target.find("attribute")
                    if box is None or attr is None:
                        continue
                    vtype = attr.get("vehicle_type", "")
                    mapped = CLASS_MAP.get(vtype)
                    if mapped is None:
                        continue
                    class_id = CLASS_NAMES.index(mapped)
                    left, top = float(box.get("left")), float(box.get("top"))
                    w, h = float(box.get("width")), float(box.get("height"))
                    lines.append(yolo_line(class_id, left, top, w, h, img_w, img_h))
                    class_counts[mapped] += 1
                    n_boxes += 1

            out_stem = f"{seq_name}_{frame_num:05d}"
            copyfile(src_img, images_out / f"{out_stem}.jpg")
            (labels_out / f"{out_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_images += 1

    return n_images, n_boxes, class_counts, skipped_sequences


def main():
    if not DETRAC_ROOT.exists():
        raise SystemExit(
            f"UA-DETRAC not found at {DETRAC_ROOT}\n"
            "Set the UA_DETRAC_ROOT environment variable if you downloaded it elsewhere."
        )

    summary = {}
    for split_name, (xml_subdir, out_name) in SPLITS.items():
        xml_dir = DETRAC_ROOT / xml_subdir
        if not xml_dir.exists():
            print(f"WARNING: {xml_dir} not found, skipping {split_name}")
            continue
        n_images, n_boxes, class_counts, skipped = convert_split(xml_dir, out_name)
        summary[split_name] = (n_images, n_boxes, class_counts, skipped)
        print(f"[{split_name}] {n_images} images, {n_boxes} boxes, classes={class_counts}")
        if skipped:
            print(f"[{split_name}] skipped {len(skipped)} sequences (images not found): "
                  f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    data_yaml = OUT_ROOT / "data.yaml"
    data_yaml.write_text(
        "path: " + str(OUT_ROOT).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names: " + str(CLASS_NAMES).replace("'", '"') + "\n",
        encoding="utf-8",
    )

    print(f"\ndata.yaml written to {data_yaml}")
    print("NOTE: 'motorcycle' has 0 examples -- UA-DETRAC has no motorcycle class. "
          "Still need supplementary motorcycle images (Report Sec. VI.A) before "
          "training a model that can actually detect motorcycles.")


if __name__ == "__main__":
    main()
