"""Train the baseline + every experiment in experiments.yaml, then score each best.pt on the TEST split.

Usage:
  python src/train.py                       # all experiments
  python src/train.py --only B0_baseline E2_no_aug
Outputs: runs/<name>/ (Ultralytics curves, results.csv, weights) and one row per experiment in results/summary.csv.
"""
import argparse
import csv
import time
from pathlib import Path

import yaml

SUMMARY = Path("results/summary.csv")


def upsert(row, path=SUMMARY):
    """Replace this experiment's row (re-runs overwrite), keep the others."""
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    rows = [r for r in rows if r["experiment"] != row["experiment"]] + [row]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments.yaml")
    ap.add_argument("--only", nargs="*", help="experiment names to run (default: all)")
    a = ap.parse_args()
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(a.config))
    for name, exp in cfg["experiments"].items():
        if a.only and name not in a.only:
            continue
        params = {**cfg["common"], **(exp or {})}
        desc = params.pop("desc", "")
        model = YOLO(params.pop("model"))
        t0 = time.time()
        model.train(project="runs", name=name, exist_ok=True, **params)
        train_min = (time.time() - t0) / 60

        weights = Path("runs") / name / "weights" / "best.pt"
        best = YOLO(str(weights))
        m = best.val(data=params["data"], split="test", imgsz=params.get("imgsz", 640), batch=1,
                     project="runs", name=f"{name}_test", exist_ok=True)  # batch=1 -> per-image latency
        row = {"experiment": name, "desc": desc,
               "test_mAP50": round(m.box.map50, 4), "test_mAP50_95": round(m.box.map, 4),
               "test_precision": round(m.box.mp, 4), "test_recall": round(m.box.mr, 4),
               "train_min": round(train_min, 1),
               "inference_ms": round(m.speed["inference"], 2),
               "params_M": round(sum(p.numel() for p in best.model.parameters()) / 1e6, 2),
               "weights_MB": round(weights.stat().st_size / 1e6, 1)}
        for i, c in enumerate(m.box.ap_class_index):
            n = best.names[int(c)]
            row[f"{n}_AP50"] = round(float(m.box.ap50[i]), 4)
            row[f"{n}_recall"] = round(float(m.box.r[i]), 4)
            row[f"{n}_precision"] = round(float(m.box.p[i]), 4)
        upsert(row)
        print(row)


if __name__ == "__main__":
    main()
