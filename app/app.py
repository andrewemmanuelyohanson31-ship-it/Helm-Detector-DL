"""Helm Detector web app. Run: streamlit run app/app.py

Image mode: detect motorcycles / helmets / riders without helmet in a photo.
Video mode: same, plus BoT-SORT tracking so each rider without a helmet is counted once.
"""
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Helm Detector", layout="wide")
DEFAULT_WEIGHTS = os.environ.get("HELM_WEIGHTS", "weights/best.pt")


@st.cache_resource
def load(path):
    return YOLO(path)


st.title("Helm Detector")
st.caption("Flags motorcycle riders without a helmet. Detections are for human review, not automatic fines.")

weights = st.sidebar.text_input("Model weights (.pt)", DEFAULT_WEIGHTS)
if not Path(weights).exists():
    st.error(f"Weights not found: {weights}. Train first (src/train.py) and copy best.pt here, or set HELM_WEIGHTS.")
    st.stop()
model = load(weights)
class_names = list(model.names.values())
default_v = next((i for i, n in enumerate(class_names) if "no" in n.lower()), 0)
violation = st.sidebar.selectbox("Violation class", class_names, index=default_v)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05,
                         help="Lower = catch more violators but more false alarms.")
mode = st.sidebar.radio("Input", ["Image", "Video"])

if mode == "Image":
    up = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png", "webp"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not read that image.")
            st.stop()
        r = model.predict(img, conf=conf, verbose=False)[0]
        counts = Counter(model.names[int(c)] for c in r.boxes.cls)
        left, right = st.columns([3, 1])
        left.image(r.plot()[:, :, ::-1], caption="Detections", use_container_width=True)
        for n in class_names:
            right.metric(n, counts.get(n, 0))
        right.metric("Inference", f"{r.speed['inference']:.0f} ms")
        if counts.get(violation):
            right.warning(f"{counts[violation]} rider(s) without helmet")
        else:
            right.success("No violation detected")
        st.dataframe([{"class": model.names[int(c)], "confidence": round(float(s), 3)}
                      for c, s in zip(r.boxes.cls, r.boxes.conf)])
else:
    up = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"])
    stride = st.sidebar.slider("Process every Nth frame", 1, 10, 2, help="Higher = faster, less smooth.")
    if up and st.button("Run"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as f:
            f.write(up.read())
        tracker_model = YOLO(weights)  # fresh model = fresh tracker state per video
        cap = cv2.VideoCapture(f.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_box, stats_box, bar = st.empty(), st.empty(), st.progress(0.0)
        ids = {n: set() for n in class_names}
        i, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % stride:
                continue
            r = tracker_model.track(frame, persist=True, tracker="botsort.yaml", conf=conf, verbose=False)[0]
            if r.boxes.id is not None:
                for c, tid in zip(r.boxes.cls.tolist(), r.boxes.id.tolist()):
                    ids[model.names[int(c)]].add(int(tid))
            frame_box.image(r.plot()[:, :, ::-1], use_container_width=True)
            stats_box.write(" | ".join(f"**{n}**: {len(s)} unique" for n, s in ids.items())
                            + f" | {i / (time.time() - t0):.1f} frames/s")
            bar.progress(min(i / total, 1.0))
        cap.release()
        os.unlink(f.name)  # uploaded footage is not kept (privacy)
        if ids[violation]:
            st.warning(f"{len(ids[violation])} unique rider(s) without helmet in this video")
        else:
            st.success("No violation detected")
