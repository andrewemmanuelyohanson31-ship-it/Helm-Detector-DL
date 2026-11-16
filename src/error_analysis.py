"""Per-image error analysis on the test split: missed objects, false alarms, class confusions.

Usage:
  python src/error_analysis.py --weights runs/B0_baseline/weights/best.pt --images data/helmet/test/images
Writes results/errors.csv and saves the worst images (annotated) to results/errors/ for manual tagging
(night, occlusion, small/far rider, cap or hood mistaken for helmet, ...).
"""
import argparse
import csv
from collections import Counter
from pathlib import Path


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, pred, thr=0.5):
    """gt: [(cls, xyxy)], pred: [(cls, conf, xyxy)].
    Greedy by confidence. A prediction overlapping an unmatched GT box (IoU >= thr) is a TP if the class
    agrees, else a confusion (gt_cls -> pred_cls). Non-overlapping predictions are FPs; leftover GT are FNs."""
    used, tp, fp, conf = set(), [], [], []
    for pc, _, pb in sorted(pred, key=lambda p: -p[1]):
        best, best_iou = None, thr
        for i, (_, gb) in enumerate(gt):
            if i not in used and (v := iou(pb, gb)) >= best_iou:
                best, best_iou = i, v
        if best is None:
            fp.append(pc)
            continue
        used.add(best)
        (tp if gt[best][0] == pc else conf).append(pc if gt[best][0] == pc else (gt[best][0], pc))
    fn = [c for i, (c, _) in enumerate(gt) if i not in used]
    return tp, fn, fp, conf


def read_gt(label_path, w, h):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
        out.append((c, ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default="data/helmet/test/images")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=30, help="how many worst images to save")
    a = ap.parse_args()

    from ultralytics import YOLO
    from prepare_data import find_images, label_for
    import cv2

    model = YOLO(a.weights)
    names = model.names
    out = Path("results/errors")
    out.mkdir(parents=True, exist_ok=True)
    rows, fn_c, fp_c, conf_c = [], Counter(), Counter(), Counter()
    for img in find_images(a.images):
        r = model.predict(str(img), conf=a.conf, verbose=False)[0]
        h, w = r.orig_shape
        gt = read_gt(label_for(img), w, h)
        pred = [(int(c), float(s), tuple(b)) for c, s, b in
                zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        tp, fn, fp, cf = match(gt, pred, a.iou)
        fn_c.update(names[c] for c in fn)
        fp_c.update(names[c] for c in fp)
        conf_c.update(f"{names[g]}->{names[p]}" for g, p in cf)
        rows.append({"image": img.name, "gt": len(gt), "tp": len(tp), "missed": len(fn), "false_alarm": len(fp),
                     "confused": len(cf), "errors": len(fn) + len(fp) + len(cf),
                     "detail": "; ".join([f"missed {names[c]}" for c in fn] + [f"false {names[c]}" for c in fp]
                                         + [f"{names[g]} as {names[p]}" for g, p in cf]),
                     "tag": ""})  # fill in by hand: night / occlusion / small / cap / blur / ...
        if rows[-1]["errors"]:
            cv2.imwrite(str(out / img.name), r.plot())

    rows.sort(key=lambda r: -r["errors"])
    keep = {r["image"] for r in rows[:a.top]}
    for f in out.iterdir():  # only keep the worst N annotated images
        if f.name not in keep:
            f.unlink()
    with open("results/errors.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("missed per class:", dict(fn_c))
    print("false alarms per class:", dict(fp_c))
    print("class confusions:", dict(conf_c))
    print(f"saved results/errors.csv and {len(keep & {p.name for p in out.iterdir()})} images in {out}")


if __name__ == "__main__":
    main()
