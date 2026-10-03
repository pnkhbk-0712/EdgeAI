"""Download the Hard Hat Workers dataset (helmet/head/person) from Roboflow (2026-10-03).

Source: Hard Hat Workers Dataset, Northeastern University - China, via Roboflow
(https://universe.roboflow.com/joseph-nelson/hard-hat-workers), License: Public Domain (CC0).
7,035 images, 3 classes: head, helmet, person. Original data via Harvard Dataverse
(doi:10.7910/DVN/7CBGOS).

Using version 10 ("raw_AllClasses"): 70/20/10 train/valid/test split, all 3 original classes
kept, raw images (no extra synthetic augmentation baked into the export) -- we'd rather apply
our own augmentation choices during training than inherit someone else's.

Requires ROBOFLOW_API_KEY in the environment (same key already used for the signs dataset in the
old project's Colab notebook -- Colab Secrets, or `export ROBOFLOW_API_KEY=...` locally).
"""
import os
from pathlib import Path

from roboflow import Roboflow

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "helmet_roboflow"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in the environment before running this script.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("joseph-nelson").project("hard-hat-workers")
    dataset = project.version(10).download("yolov8", location=str(OUT_DIR))

    print(f"Downloaded to: {dataset.location}")
    print("Next: check data.yaml for the exact class index order (head/helmet/person) -- "
          "Roboflow exports alphabetically by default, but confirm before wiring up "
          "src/demo_helmet.py's class map.")


if __name__ == "__main__":
    main()
