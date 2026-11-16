from pathlib import Path

from error_analysis import iou, match, read_gt
from prepare_data import label_for, split, stats


def test_label_for():
    assert label_for(Path("d/train/images/a.jpg")) == Path("d/train/labels/a.txt")
    assert label_for(Path("raw/a.png")) == Path("raw/a.txt")


def test_split_and_stats(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(20):
        (raw / f"{i}.jpg").write_bytes(b"x")
        if i < 18:  # two background images without labels
            (raw / f"{i}.txt").write_text("2 0.5 0.5 0.01 0.01\n1 0.5 0.5 0.3 0.3\n")
    (raw / "0.txt").write_text("0 0.1 0.1 0.2\n")  # malformed
    assert split(raw, tmp_path / "out") == {"train": 14, "val": 3, "test": 3}
    rows = stats(tmp_path / "out", ["motorcycle", "helmet", "no_helmet"])
    assert [r["split"] for r in rows] == ["test", "train", "val"]
    assert sum(r["images"] for r in rows) == 20
    assert sum(r["empty_images"] for r in rows) == 2  # image 0 is malformed, not empty
    assert sum(r["malformed_lines"] for r in rows) == 1
    assert sum(r["no_helmet"] for r in rows) == 17 and sum(r["small_boxes"] for r in rows) == 17


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_match():
    gt = [(1, (0, 0, 10, 10)), (2, (50, 50, 60, 60)), (0, (100, 100, 200, 200))]
    pred = [(1, 0.9, (0, 0, 10, 10)),        # TP
            (1, 0.8, (50, 50, 60, 61)),      # no_helmet predicted as helmet -> confusion
            (2, 0.7, (300, 300, 310, 310)),  # false alarm
            (1, 0.6, (0, 0, 10, 10))]        # duplicate of an already matched GT -> FP
    tp, fn, fp, conf = match(gt, pred)
    assert tp == [1] and fn == [0] and fp == [2, 1] and conf == [(2, 1)]


def test_read_gt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("2 0.5 0.5 0.2 0.4\nbad line\n")
    assert read_gt(p, 100, 50) == [(2, (40.0, 15.0, 60.0, 35.0))]
    assert read_gt(tmp_path / "missing.txt", 100, 50) == []
