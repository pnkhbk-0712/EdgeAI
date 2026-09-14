# UA-DETRAC — downloaded via Kaggle mirror (2026-09-15)

**Blocker resolved.** Instead of waiting on official registration at
http://detrac-db.rit.albany.edu/ (which can take days), Hieu created a Kaggle account and
used `kagglehub` to pull a public mirror: `bratjay/ua-detrac-orig`.

```python
import kagglehub
path = kagglehub.dataset_download("bratjay/ua-detrac-orig")
```

Auth note: kagglehub's default flow tries to open a browser for OAuth, which hangs forever in
a headless/CLI environment with no browser. Fixed by using Kaggle's newer token auth instead —
saved the token to `~/.kaggle/access_token` (not committed anywhere; this lives outside the
repo, in the user's home directory, same as SSH keys).

## Real numbers (verified after download, not from the dataset's own description)

- **12 GB** total
- **140,131** JPEG frames
- **100** XML annotation files (one per `MVI_#####` sequence — this is UA-DETRAC's native
  per-sequence annotation format, covering vehicle bounding boxes + attributes like
  occlusion/truncation across the sequence's frames)
- Downloaded to `D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\.cache\kagglehub\datasets\bratjay\ua-detrac-orig\versions\2`
  — **not** the C: drive default. kagglehub's default cache is `~/.cache/kagglehub`, which on
  this machine resolves under `C:\Users\ADMIN\` — moved off C: after the fact and now pinned
  via the `KAGGLEHUB_CACHE` environment variable (see below) so it never lands there again.
  Outside the git repo entirely either way — nothing to gitignore, but also nothing that
  travels with `git clone`; each teammate who needs it re-runs the download locally.

  ```bash
  export KAGGLEHUB_CACHE="D:/DOANMINHHIEU/STUDIES/Ky9/01_Ai/.cache/kagglehub"
  ```

  Also: the Kaggle API token is passed as the `KAGGLE_API_TOKEN` environment variable per
  command rather than written to `~/.kaggle/access_token` — avoids a credential file on disk
  at all, not just avoids it being on C:.

## What's still open

- Haven't yet parsed the XML annotations into YOLO format (needed before this can feed
  `model.train(...)` the way the coco8 smoke-test did in `docs/IMPLEMENTATION_REPORT.md`).
- UA-DETRAC has no motorcycle class (confirmed earlier from the paper/dataset description) —
  still need the supplementary motorcycle images per Report Sec. VI.A.
- 12GB is large for a laptop to iterate on quickly — worth deciding whether to train on the
  full set or a stratified subset of sequences first.
