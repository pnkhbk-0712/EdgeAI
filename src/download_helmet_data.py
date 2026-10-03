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
import shutil
from pathlib import Path

from roboflow import Roboflow

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "helmet_roboflow"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in the environment before running this script.")

    # IMPORTANT: do NOT pre-create OUT_DIR. The roboflow SDK's .download(location=...) treats an
    # already-existing target directory as "already downloaded" and silently skips writing any
    # files there (no error, no warning) -- bit us once already (2026-10-03, empty folder from a
    # prior failed attempt caused a real download to silently no-op). If OUT_DIR exists from a
    # previous run, wipe it first so the SDK actually downloads into it.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("joseph-nelson").project("hard-hat-workers")
    dataset = project.version(10).download("yolov8", location=str(OUT_DIR))

    # Verify the download actually wrote something -- don't just trust the SDK's return value
    # after getting burned by the silent-skip behavior above.
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"Download reported success but {data_yaml} doesn't exist -- something is still "
            "wrong (check Roboflow account access to this dataset, API key validity, or "
            "quota). Do not proceed to training until this file is confirmed present."
        )

    print(f"Downloaded to: {dataset.location}")
    print(f"Confirmed: {data_yaml} exists.")
    print("Next: check data.yaml for the exact class index order (head/helmet/person) -- "
          "Roboflow exports alphabetically by default, but confirm before wiring up "
          "src/demo_helmet.py's class map.")


if __name__ == "__main__":
    main()
