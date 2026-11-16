"""Split a flat YOLO-format dataset into train/val/test and report data statistics.

Usage:
  python src/prepare_data.py split --src raw/ --dst data/helmet
  python src/prepare_data.py stats --root data/helmet --names motorcycle helmet no_helmet
"""
import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SMALL_AREA = (32 * 32) / (640 * 640)  # COCO "small object" threshold, at 640px input


def find_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def label_for(img):
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt; otherwise x.txt next to the image."""
    parts = list(Path(img).parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def split(src, dst, ratios=(0.7, 0.15, 0.15), seed=42):
    assert abs(sum(ratios) - 1) < 1e-6, "ratios must sum to 1"
    imgs = find_images(src)
    if not imgs:
        raise SystemExit(f"no images found in {src}")
    dupes = [n for n, c in Counter(p.name for p in imgs).items() if c > 1]
    if dupes:
        raise SystemExit(f"duplicate file names would overwrite each other: {dupes[:5]}")
    random.Random(seed).shuffle(imgs)
    n_train, n_val = int(len(imgs) * ratios[0]), int(len(imgs) * ratios[1])
    groups = {"train": imgs[:n_train], "val": imgs[n_train:n_train + n_val], "test": imgs[n_train + n_val:]}
    for name, files in groups.items():
        for sub in ("images", "labels"):
            (Path(dst) / name / sub).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, Path(dst) / name / "images" / img.name)
            lbl = label_for(img)
            if lbl.exists():  # no label file = background image (no objects)
                shutil.copy2(lbl, Path(dst) / name / "labels" / lbl.name)
    return {k: len(v) for k, v in groups.items()}


def stats(root, names):
    """One row per split folder (any subfolder of root with an images/ dir, e.g. Roboflow's 'valid')."""
    rows = []
    for split_dir in sorted(d for d in Path(root).iterdir() if (d / "images").is_dir()):
        cls, empty, small, boxes, bad = Counter(), 0, 0, 0, 0
        imgs = find_images(split_dir / "images")
        for img in imgs:
            lbl = label_for(img)
            lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            if not lines:
                empty += 1
            for l in lines:
                if len(l) != 5:  # polygon/segmentation labels or junk
                    bad += 1
                    continue
                c, w, h = int(l[0]), float(l[3]), float(l[4])
                cls[c] += 1
                boxes += 1
                small += w * h < SMALL_AREA
        row = {"split": split_dir.name, "images": len(imgs), "empty_images": empty,
               "boxes": boxes, "small_boxes": small, "malformed_lines": bad}
        for i in sorted(set(cls) | set(range(len(names)))):
            row[names[i] if i < len(names) else f"unknown_class_{i}"] = cls[i]
        rows.append(row)
    return rows


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--src", required=True)
    s.add_argument("--dst", default="data/helmet")
    s.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    s.add_argument("--seed", type=int, default=42)
    t = sub.add_parser("stats")
    t.add_argument("--root", default="data/helmet")
    t.add_argument("--names", nargs="+", default=["motorcycle", "helmet", "no_helmet"])
    t.add_argument("--out", default="results/data_stats.csv")
    a = ap.parse_args()
    if a.cmd == "split":
        print(split(a.src, a.dst, a.ratios, a.seed))
    else:
        rows = stats(a.root, a.names)
        write_csv(rows, a.out)
        for r in rows:
            print(r)
        print(f"saved {a.out}")
