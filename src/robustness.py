"""Robustness test: re-score a model on corrupted copies of the test split (blur, night, noise, low JPEG).

Usage: python src/robustness.py --weights runs/B0_baseline/weights/best.pt
Output: results/robustness.csv (one row per condition, 'clean' included for reference).
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

PERTURB = {
    "clean": lambda im: im,
    "motion_blur": lambda im: cv2.filter2D(im, -1, np.eye(15, dtype=np.float32) / 15),  # moving rider/camera
    "night": lambda im: np.clip(im.astype(np.float32) * 0.35, 0, 255).astype(np.uint8),
    "sensor_noise": lambda im: np.clip(im + np.random.default_rng(0).normal(0, 20, im.shape), 0, 255).astype(np.uint8),
    "low_jpeg": lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 15])[1], 1),  # CCTV
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/helmet.yaml")
    ap.add_argument("--test", default="data/helmet/test")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    from ultralytics import YOLO
    from prepare_data import find_images

    names = yaml.safe_load(open(a.data))["names"]
    model = YOLO(a.weights)
    rows = []
    for cond, fn in PERTURB.items():
        root = Path("data/robust") / cond
        (root / "images").mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(a.test) / "labels", root / "labels", dirs_exist_ok=True)
        for img in find_images(Path(a.test) / "images"):
            cv2.imwrite(str(root / "images" / img.name), fn(cv2.imread(str(img))))
        ds = root.parent / f"{cond}.yaml"
        ds.write_text(yaml.safe_dump({"path": str(root.resolve()), "train": "images", "val": "images",
                                      "names": names}))
        m = model.val(data=str(ds), imgsz=a.imgsz, project="runs", name=f"robust_{cond}", exist_ok=True)
        row = {"condition": cond, "mAP50": round(m.box.map50, 4), "mAP50_95": round(m.box.map, 4),
               "precision": round(m.box.mp, 4), "recall": round(m.box.mr, 4)}
        for i, c in enumerate(m.box.ap_class_index):
            row[f"{model.names[int(c)]}_recall"] = round(float(m.box.r[i]), 4)
        rows.append(row)
        print(row)
    Path("results").mkdir(exist_ok=True)
    with open("results/robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print("saved results/robustness.csv")


if __name__ == "__main__":
    main()
