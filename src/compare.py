"""Overlay training curves of all experiments (for overfitting / convergence analysis).

Usage: python src/compare.py            -> results/curves.png, results/fit_gap.csv
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PANELS = [("train/box_loss", "val/box_loss", "Box loss (solid=train, dashed=val)"),
          ("train/cls_loss", "val/cls_loss", "Class loss (solid=train, dashed=val)"),
          ("metrics/mAP50(B)", None, "Val mAP@50"),
          ("metrics/mAP50-95(B)", None, "Val mAP@50-95")]


def main():
    runs = sorted(p for p in Path("runs").glob("*/results.csv") if not p.parent.name.endswith("_test"))
    if not runs:
        raise SystemExit("no runs/*/results.csv yet - train first")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    gaps = []
    for path in runs:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # older Ultralytics pads column names
        name = path.parent.name
        for ax, (a, b, title) in zip(axes.flat, PANELS):
            line, = ax.plot(df["epoch"], df[a], label=name)
            if b:
                ax.plot(df["epoch"], df[b], "--", color=line.get_color())
            ax.set_title(title)
            ax.set_xlabel("epoch")
        last = df.iloc[-1]
        best = df["metrics/mAP50(B)"].idxmax()
        gaps.append({"experiment": name,
                     "final_train_box_loss": round(last["train/box_loss"], 4),
                     "final_val_box_loss": round(last["val/box_loss"], 4),
                     "gap_val_minus_train": round(last["val/box_loss"] - last["train/box_loss"], 4),
                     "best_epoch": int(df.loc[best, "epoch"]),
                     "best_val_mAP50": round(df.loc[best, "metrics/mAP50(B)"], 4)})
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/curves.png", dpi=150)
    with open("results/fit_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gaps[0]))
        w.writeheader()
        w.writerows(gaps)
    print("saved results/curves.png and results/fit_gap.csv")


if __name__ == "__main__":
    main()
