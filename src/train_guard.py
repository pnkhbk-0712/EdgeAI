"""Run a command (e.g. `yolo detect train ...`) and stop it if the GPU gets too hot.

Polls nvidia-smi every POLL_SEC; two consecutive reads above LIMIT_C terminate the run (Ultralytics keeps
last.pt each epoch, so `yolo detect train resume model=<run>/weights/last.pt` continues it). Temperatures are
appended to GUARD_LOG. Usage: python src/train_guard.py [--limit 85] -- yolo detect train data=... device=0
"""
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

POLL_SEC = 30
GUARD_LOG = Path("D:/DOANMINHHIEU/STUDIES/Ky9/01_Ai/_tools/gpu_temps.csv")


def gpu_temp_c():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
    return int(out[0])


ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=85)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
args = ap.parse_args()
cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
if not cmd:
    raise SystemExit("give a command after --")

proc = subprocess.Popen(cmd)
hot = 0
try:
    while proc.poll() is None:
        time.sleep(POLL_SEC)
        t = gpu_temp_c()
        with GUARD_LOG.open("a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')},{t}\n")
        hot = hot + 1 if t > args.limit else 0
        if hot >= 2:
            print(f"[train_guard] GPU {t} C > {args.limit} C twice -> stopping; resume later from weights/last.pt")
            proc.terminate()
            proc.wait(timeout=60)
            raise SystemExit(2)
finally:
    if proc.poll() is None:
        proc.terminate()
raise SystemExit(proc.returncode)
