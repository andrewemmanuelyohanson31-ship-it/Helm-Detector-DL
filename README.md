# Helm Detector — Motorcycle Helmet Compliance Detection with Deep Learning

COMP6826001 Deep Learning · Final Project · BINUS University · 2026/2027

A computer-vision system that detects **motorcycles, riders wearing a helmet, and riders without a helmet**
in road images and traffic video. It uses a fine-tuned **YOLO11** object detector (PyTorch / Ultralytics) and
**BoT-SORT** tracking so that each violating rider is counted once. Results appear in a Streamlit web app.

Inspired by *"Real-Time Helmet Violation Detection using YOLOv11 + BoT-SORT"* (Uzair Ahmad Khan, LinkedIn).
That post used YOLO11n, three classes (Bike / Helmet / No-Helmet), BoT-SORT tracking and an on-screen count
dashboard. It reported no metrics. This project adds what the course requires on top: a documented dataset,
a baseline, controlled experiments, training-curve analysis, error analysis, robustness tests and a
real-world suitability evaluation.

> **Source documents.** Requirements come from two files in `D:\Binus\deeplearn`:
> **(A)** `FINAL_PROJECT_COMP6826001_Deep_Learning.docx` (2026/2027, the **current, authoritative** brief) and
> **(B)** `COMP6826001 Deep Learning_Final Project.docx` (2025/2026, older brief: proposal checkpoint, demo video,
> report format). If A and B disagree, follow A and confirm with the lecturer (see [§11](#11-open-questions-for-the-lecturer)).

---

## Contents
1. [Project proposal](#1-project-proposal)
2. [Requirement §6 — Deep Learning Model Requirement (full)](#2-requirement-6--deep-learning-model-requirement-full)
3. [Everything the brief requires → where we satisfy it](#3-requirements-checklist)
4. [Framework: end-to-end pipeline](#4-framework-end-to-end-pipeline)
5. [Experiment plan](#5-experiment-plan)
6. [Evaluation, analysis & real-world suitability](#6-evaluation-analysis--real-world-suitability)
7. [Application](#7-application)
8. [Things needed (resources)](#8-things-needed-resources)
9. [How to run](#9-how-to-run)
10. [Team, timeline & AI usage log](#10-team-timeline--ai-usage-log)
11. [Open questions for the lecturer](#11-open-questions-for-the-lecturer)
12. [References](#12-references)

---

## 1. Project proposal

*Covers brief B "Project Proposal (Checkpoint)" (title, problem, dataset + licence, model/baseline plan,
metrics + success criteria, references) and brief A §4 (target user, problem, proposed solution, usage scenario).*

### 1.1 Title
**Helm Detector: Real-Time Detection of Motorcycle Riders Without Helmets Using Fine-Tuned YOLO11 and BoT-SORT Tracking**

### 1.2 Problem statement (brief A §4)
| Question | Answer |
|---|---|
| **Who has the problem? (Target user / stakeholder)** | Traffic police and road-safety agencies (e.g. Korlantas Polri's camera-based ETLE enforcement), city transport departments, campus or estate security, and road-safety researchers who count compliance rates. Indirectly, motorcyclists: the riders most at risk. |
| **What is the problem?** | Motorcycles make up most of Indonesia's road traffic, and riders without a helmet face a much higher risk of fatal head injury. Wearing a helmet is mandatory (UU No. 22/2009 on Traffic and Road Transport, Art. 106(8) and Art. 291), but enforcement depends on officers watching the road or hours of CCTV by hand. That is slow, costly, inconsistent, and can't scale to thousands of cameras. |
| **Why is Deep Learning appropriate?** | The task is visual object detection in cluttered, changing scenes: many riders per frame, small and distant heads, occlusion, varied lighting and weather, caps and hoods that look like helmets. Hand-crafted features (HOG, colour or shape rules) break under these conditions. CNN-based detectors learn these features from labelled data and run in real time on a GPU. Transfer learning from COCO (which already contains *person* and *motorcycle*) helps when labelled helmet data is limited. |
| **Proposed solution** | Fine-tune a YOLO11 detector on 3 classes (`motorcycle`, `helmet`, `no_helmet`). Add BoT-SORT multi-object tracking for video so each violator is counted once, not once per frame. Serve it in a web app that takes a photo or video and returns annotated detections, per-class counts and violation flags. |
| **Usage scenario** | (1) **Offline review:** an officer or researcher uploads CCTV clips or photos from an intersection, and the system shortlists frames with violators for **human review**. (2) **Compliance survey:** count the helmet-use rate at a location for a road-safety study. (3) **Live monitoring (stretch goal):** a fixed roadside camera in daylight or under street lighting. Out of scope: automatic fines without a human check, licence-plate reading, identifying individuals. |

### 1.3 Objectives & scope
- **O1** Build a reproducible pipeline: data preparation → training → evaluation → analysis.
- **O2** Fine-tune YOLO11 and compare it with a from-scratch baseline and ≥ 2 other controlled experiments.
- **O3** Reach the success criteria in §1.6 on a held-out test set.
- **O4** Measure robustness (night, blur, noise, CCTV compression) and computational cost (latency, size, FPS).
- **O5** Deliver a working app (image + video) and test it with at least 3–5 target-style users.

**Scope:** still images and pre-recorded video, daytime and night, front, rear and side views of riders.
**Out of scope:** plate recognition, face recognition, helmet *quality* (SNI or not), chin-strap detection.

### 1.4 Dataset (source, description, licence) — brief A §5, brief B "Dataset Policy"
Planned approach: **one public primary dataset + one small self-collected local test set.**

| Candidate | Classes | Size / format | Licence | Role |
|---|---|---|---|---|
| **Roboflow Universe** helmet datasets (search "motorcycle helmet" / "bike helmet detection") | usually motorbike/rider, helmet, no-helmet | Hundreds to thousands of images; exports **YOLO11 format** directly | Shown on each dataset page (often CC BY 4.0). **Record the exact URL, version and licence.** | **Primary** (train/val/test) |
| Kaggle "Helmet Detection" (andrewmvd) | With Helmet, Without Helmet | 764 images, Pascal VOC XML | See the Kaggle page (check before use) | Alternative or extra data; convert VOC → YOLO with Roboflow |
| HELMET dataset, Siebert & Lin (2020), Myanmar road video | rider position + helmet use | Large, annotated video frames | Per the paper's data release (check) | Optional: **cross-domain** robustness test |
| **Self-collected** Jakarta/Tangerang road photos and video (team phones, public roads) | same 3 classes | Target 150–300 frames, labelled in Roboflow/CVAT | Our own; collected in public, faces and plates blurred before publishing | **Real-world held-out test** (domain shift: Indonesian roads) |

What we must report for the chosen dataset (A §5): source · characteristics (resolution, viewpoint, day/night)
· number of images and boxes · labels/classes · **class distribution** (`src/prepare_data.py stats`) ·
data quality (empty or malformed labels, duplicates) · **potential bias** (country, camera angle, helmet
styles, lighting, few night images) · **limitations** (e.g. `no_helmet` is a minority class → imbalance).

**Data preparation** (A §5): check and clean labels (malformed lines, wrong class ids, duplicate images)
· remap class ids to `motorcycle/helmet/no_helmet` · resize to 640 px with letterbox (done by YOLO) ·
pixel normalisation to [0,1] (done by YOLO) · augmentation (mosaic, flip, HSV, scale/translate; tested in E2)
· **70/15/15 train/val/test split**, seed 42. **Leakage warning:** if the dataset contains consecutive video
frames, split **by video/source** rather than by frame, or near-duplicate frames will inflate the test score.

### 1.5 Model / methodology & baseline
- **Model:** YOLO11 (Ultralytics, PyTorch), one-stage anchor-free detector. Details in [§2](#2-requirement-6--deep-learning-model-requirement-full).
- **Baseline (B0):** YOLO11n, COCO-pretrained, full fine-tune, default augmentation, SGD lr0 = 0.01, 50 epochs, 640 px, batch 16.
- **Experiments:** from scratch (E1), no augmentation (E2), larger YOLO11s (E3), plus optional AdamW (E4) and frozen backbone (E5). See [§5](#5-experiment-plan).
- **Tracking:** BoT-SORT (Ultralytics built-in `botsort.yaml`) for unique-violator counting in video.

### 1.6 Evaluation metrics & success criteria
| Metric | Why this metric (A §11: chosen for the problem, not popularity) | Target |
|---|---|---|
| **Recall of `no_helmet`** | Main metric. A missed violator is the costliest error for an enforcement shortlist. | ≥ 0.85 |
| **Precision of `no_helmet`** | False accusations waste reviewers' time and hurt trust. | ≥ 0.80 |
| **mAP@50** (all classes) | Standard detection quality at a loose IoU. | ≥ 0.80 |
| **mAP@50-95** | Box tightness; matters for small, distant heads. | report |
| Confusion matrix (`helmet` ↔ `no_helmet`) | Shows the errors that matter most. | report |
| Inference latency / FPS, params, weights size, train time | Deployment cost (A §15). | ≥ 15 FPS on a T4 GPU at 640 px |
| Drop in mAP@50 under night/blur/noise/JPEG | Robustness (A §15). | report; < 25 % relative drop |
| User testing: task success + SUS-style usability score | Practical evaluation (A §17). | ≥ 3–5 users |

Targets are goals, not results. Report the real numbers you get, including any misses.

---

## 2. Requirement §6 — Deep Learning Model Requirement (full)

**Brief A §6 (translated from Indonesian):**
> Students **must build at least one Deep Learning model** using modern programming tools/frameworks.
>
> **Allowed frameworks include:** PyTorch · TensorFlow · Keras · JAX · or another relevant deep learning framework.
>
> **The model may be:** CNN · RNN/LSTM/GRU · Transformer · Autoencoder · MLP · Vision Transformer ·
> pretrained model · or another suitable architecture.

**How this project satisfies §6:**

| §6 requirement | This project |
|---|---|
| At least one DL model | **YOLO11** object detector (YOLO11n baseline + YOLO11s variant), plus a from-scratch YOLO11n for comparison. |
| Modern framework | **PyTorch** through the **Ultralytics** library (`ultralytics>=8.3`). |
| Model type | **CNN-based detector with an attention block** (C2PSA), started from a **pretrained model** (COCO) and fine-tuned. Allowed under "CNN" and "pretrained model". |

Since §6 allows a pretrained model, §7 (Model Construction) and §8 (Pretrained Model) also apply. §8 says
*"Load pretrained → run inference → show result" does NOT meet the requirement.* We fine-tune and run
controlled experiments, and the report must explain the following:

**§7 Model construction — must appear in the report:**
| Item | YOLO11 (fill in the final values used) |
|---|---|
| Architecture | **Backbone:** Conv + **C3k2** blocks → **SPPF** (spatial pyramid pooling) → **C2PSA** (position-sensitive attention). **Neck:** PAN/FPN-style top-down + bottom-up feature fusion. **Head:** decoupled, anchor-free detection head at 3 scales (stride 8/16/32). |
| Input | RGB image, letterboxed to 640×640, pixels scaled to [0,1]. |
| Output | N boxes × (x, y, w, h, class scores for 3 classes) → NMS → final boxes + confidence + class. With tracking, also a track ID. |
| Important components | C3k2 (efficient CSP bottleneck), SPPF (multi-scale context), C2PSA (attention for small, occluded objects), DFL box regression. |
| Loss function | **Box:** CIoU loss + Distribution Focal Loss (DFL). **Class:** Binary Cross-Entropy. Total = 7.5·box + 0.5·cls + 1.5·dfl (Ultralytics defaults). |
| Optimizer | SGD (momentum 0.937, weight decay 5e-4) for the baseline; AdamW in E4. |
| Learning rate | lr0 = 0.01, warm-up 3 epochs, linear decay to lr0·0.01. |
| Batch size | 16 |
| Epochs | 50 (no early stopping, so all curves are comparable) |
| Activation | SiLU |
| Training procedure | Mosaic augmentation (switched off for the last 10 epochs), AMP mixed precision, best checkpoint picked by val fitness (0.1·mAP50 + 0.9·mAP50-95). |
| **Why chosen** | Real-time speed on modest hardware (deployable on roadside or edge devices), strong small-object detection, mature tooling, same family as the reference project. n (nano) is the speed baseline; s tests whether more capacity is worth the cost. |

**§8 Pretrained model — must appear in the report:**
| Item | Answer |
|---|---|
| Architecture | As above (YOLO11n / YOLO11s). |
| Pretraining objective | Supervised object detection (box regression + classification). |
| Pretraining dataset/domain | **MS COCO** (80 classes, ~118k training images). Includes `person` and `motorcycle` but **no helmet class**. |
| Transfer-learning strategy | Replace the 80-class head with a 3-class head and **fine-tune all layers** (B0). E5 compares freezing the backbone; E1 compares no pretraining. |
| Layers/components used | Backbone + neck reused from COCO; the detection head's classification layers are re-initialised for 3 classes. |
| Why appropriate | COCO features for people and motorcycles transfer directly, which reduces the labelled data we need. |
| Limitations | COCO is mostly daytime, Western, consumer photos, a domain gap with Indonesian CCTV at night. The model may confuse caps, hoods or hair with helmets. Licence: Ultralytics is **AGPL-3.0** (fine for coursework; keep the repo AGPL-compatible if it is made public or deployed). |

---

## 3. Requirements checklist

Tick these off as the project progresses. "Where" points to the evidence.

### Brief A (2026/2027): working system + technical report
| § | Requirement | Where / how | Done |
|---|---|---|---|
| 1 | End-to-end: problem → data → model → training → evaluation → experiments → analysis → system → real-world evaluation | [§4](#4-framework-end-to-end-pipeline) pipeline | ☐ |
| 4 | Real-world problem: target user, problem, solution, usage scenario | [§1.2](#12-problem-statement-brief-a-4) | ☐ |
| 5 | Dataset: source, characteristics, count, labels, distribution, quality, bias, limitations | [§1.4](#14-dataset-source-description-licence--brief-a-5-brief-b-dataset-policy), `results/data_stats.csv` | ☐ |
| 5 | Data preparation + train/val/test split | `src/prepare_data.py` | ☐ |
| 6 | ≥ 1 DL model in a modern framework | YOLO11 / PyTorch, [§2](#2-requirement-6--deep-learning-model-requirement-full) | ☐ |
| 7 | Explain architecture, I/O, layers, loss, optimizer, LR, batch, epochs, activation, procedure + why | [§2](#2-requirement-6--deep-learning-model-requirement-full) table → report ch. 3 | ☐ |
| 8 | Pretrained model is not a black box (objective, dataset, fine-tune strategy, limitations) | [§2](#2-requirement-6--deep-learning-model-requirement-full) | ☐ |
| 9 | Baseline + ≥ 2 experimental scenarios | `experiments.yaml` (B0 + E1–E3 required, E4–E5 optional) | ☐ |
| 10 | Each experiment: what changed, why, what effect | [§5](#5-experiment-plan) table → report ch. 5–7 | ☐ |
| 11 | Task-appropriate metrics (P, R, mAP) | `results/summary.csv` | ☐ |
| 12 | Training analysis: train/val loss + metric curves; over/underfitting, convergence, imbalance, hyperparameter sensitivity | `src/compare.py` → `results/curves.png`, `fit_gap.csv` | ☐ |
| 13 | Systematic model comparison + trade-offs (accuracy vs time vs params) | `results/summary.csv` | ☐ |
| 14 | Error analysis with patterns | `src/error_analysis.py` → `results/errors.csv` (+ manual `tag` column) | ☐ |
| 15 | Real-world suitability: performance, robustness, compute cost, scalability, limitations, practical suitability | `src/robustness.py` + [§6](#6-evaluation-analysis--real-world-suitability) | ☐ |
| 16 | Working app: input → inference with OUR trained model → output shown to user (no mockups) | `app/app.py` | ☐ |
| 17 | User testing / practical evaluation | [§6.4](#64-user-testing-brief-a-17) | ☐ |
| 18 | Technical report with the 10-chapter structure | [§6.5](#65-technical-report-outline-brief-a-18) | ☐ |
| 19–22 | AI usage declared, logged, verified; **no fabricated results, data or feedback** | [§10.3](#103-ai-usage-log-brief-a-20) | ☐ |
| 25 | Success criteria (all 11 ✓ items) | this table | ☐ |

### Brief B (2025/2026): items to confirm are still required
| Item | Requirement | Done |
|---|---|---|
| Proposal checkpoint (Week 6, 1–2 pages) | Title + problem, dataset + licence, model/baseline plan, metrics + success criteria, ≥ 2–3 references → [§1](#1-project-proposal) | ☐ |
| Demo video | < 5 min, MP4 or YouTube: problem, workflow, results, professional narration | ☐ |
| Report length / format | 10–15 pages, academic style, IEEE/APA references, appendix with team contribution, screenshots, code snippets | ☐ |
| Team contribution statement | Per-member contributions + how we coordinated (Git, issues), contribution log if requested | ☐ |
| Ethics & responsible AI | Bias, explainability, privacy, no harmful use → [§6.6](#66-ethics--responsible-ai) | ☐ |
| Submission | GitHub repo (this one) + PDF report + video link via Binusmaya by the Week 14 deadline | ☐ |

---

## 4. Framework: end-to-end pipeline

```
REAL PROBLEM  riders without helmets; enforcement is manual
     ↓
DATA          public helmet dataset (+ own local test set) → clean, remap classes → stats → 70/15/15 split
     ↓        src/prepare_data.py
MODEL         YOLO11n (COCO-pretrained) → 3-class head
     ↓
TRAINING      B0 baseline + E1..E5, same seed/epochs, one variable changed per run
     ↓        src/train.py + experiments.yaml        → runs/<exp>/  (curves, weights)
EVALUATION    each best.pt scored on the TEST split  → results/summary.csv
     ↓
ANALYSIS      training curves / overfitting gap     → src/compare.py        → results/curves.png
     ↓        error patterns                        → src/error_analysis.py → results/errors.csv
ROBUSTNESS    night / blur / noise / CCTV JPEG      → src/robustness.py     → results/robustness.csv
     ↓
SYSTEM        Streamlit app: image + video, BoT-SORT unique-violator count   → app/app.py
     ↓
REAL-WORLD    user testing + local Indonesian test set + cost analysis
     ↓
AI REFLECTION AI usage log + verification notes
```

### Repository layout
```
Helm-Detector-DL/
├── README.md              ← this file (requirements, proposal, plan)
├── experiments.yaml       ← baseline + experiment configs (one variable each)
├── requirements.txt
├── data/helmet.yaml       ← dataset paths + class names (dataset itself is git-ignored)
├── src/
│   ├── prepare_data.py    ← split + data statistics
│   ├── train.py           ← train all experiments, test-set metrics → results/summary.csv
│   ├── compare.py         ← overlay curves, overfitting gap
│   ├── error_analysis.py  ← per-image misses / false alarms / confusions
│   └── robustness.py      ← corrupted test sets
├── app/app.py             ← Streamlit app
├── tests/test_core.py     ← unit tests for data split/stats and error matching
└── results/               ← CSVs and plots that go into the report (commit these)
```

---

## 5. Experiment plan

All runs share seed 42, 50 epochs, 640 px, batch 16 and the same split. **Only one variable changes per experiment.**

| ID | Variable (A §10) | Baseline value → experiment value | Why (hypothesis) | Measure |
|---|---|---|---|---|
| **B0** | — | YOLO11n, COCO-pretrained, SGD 0.01, default augmentation | Reference point | all |
| **E1** | Transfer learning | pretrained → **from scratch** | Expect much lower mAP and slower convergence on a small dataset; shows what COCO pretraining contributes | mAP, curves |
| **E2** | Data augmentation | default aug → **none** | Expect faster overfitting (widening train/val loss gap) and worse robustness | fit gap, robustness |
| **E3** | Architecture size | YOLO11n → **YOLO11s** | More capacity may help small, distant heads but costs latency and size | mAP vs ms/params |
| E4 *(opt.)* | Optimizer + LR | SGD 0.01 → **AdamW 0.001** | Sensitivity to the optimizer and learning rate | curves, mAP |
| E5 *(opt.)* | Fine-tuning strategy | full fine-tune → **freeze backbone** | Faster training; does the domain gap need backbone adaptation? | mAP, train time |

**Research questions** (report ch. 5):
RQ1 How much does COCO pretraining improve helmet-violation detection on a small dataset? (B0 vs E1)
RQ2 Does augmentation reduce overfitting and improve robustness? (B0 vs E2)
RQ3 Is a larger model worth its extra compute for this deployment? (B0 vs E3)
RQ4 Is the best model good enough, and robust enough, for real roadside use? (§6)

Free GPUs (Kaggle T4/P100, Colab T4): YOLO11n for 50 epochs on ~1–3k images takes roughly 10–40 min per run.
Budget about 3–6 GPU-hours in total and record the actual time in `summary.csv` (`train_min`).

---

## 6. Evaluation, analysis & real-world suitability

### 6.1 Training analysis (A §12)
`python src/compare.py` overlays **train and val box/cls loss** and **val mAP@50 / mAP@50-95** for every experiment
(`results/curves.png`) and writes the final **val − train loss gap** and best epoch (`results/fit_gap.csv`).
Discuss: overfitting (val loss rises while train loss falls; expect it in E2), underfitting or slow
convergence (E1), instability (loss spikes; check E4), class imbalance (`no_helmet` AP vs `helmet` AP),
and hyperparameter sensitivity (B0 vs E4).

### 6.2 Error analysis (A §14)
`python src/error_analysis.py --weights runs/<best>/weights/best.pt` → `results/errors.csv` (per-image misses,
false alarms and confusions such as `no_helmet as helmet`) plus the 30 worst annotated images in `results/errors/`.
**Manually fill the `tag` column** (night, occlusion/pillion, small/far, cap or hood, blur, crowd), then
summarise the patterns, e.g. *"most missed no_helmet cases are pillion passengers occluded by the driver"*.

### 6.3 Real-world suitability (A §15)
| Aspect | Evidence |
|---|---|
| Performance | `summary.csv` test metrics, especially `no_helmet_recall` / `no_helmet_precision` |
| Robustness | `robustness.csv` (clean vs motion blur, night, sensor noise, low JPEG) + the self-collected Indonesian test set (domain shift) |
| Computational cost | `train_min`, `inference_ms`, `params_M`, `weights_MB`; FPS shown live in the app; note GPU vs CPU |
| Scalability | Frames/s per GPU → how many camera streams one GPU can serve; batch or offline processing for archives |
| Limitations | Night, heavy occlusion, pillion riders, caps vs helmets, camera angle, dataset country bias |
| Practical suitability | Is the most accurate model (E3?) the best to *deploy*, or does YOLO11n's speed win? Human-in-the-loop review required |

### 6.4 User testing (brief A §17)
3–5 participants (e.g. a campus security officer, classmates acting as reviewers, someone with traffic or
road-safety experience). Each one uploads 3 images and 1 video, interprets the output, and completes a short
questionnaire: task success, time taken, ease of use (1–5), trust in the result (1–5), would use it (Y/N),
free-text feedback. **Record real responses only.** Fabricated feedback is explicitly forbidden (A §21).

### 6.5 Technical report outline (brief A §18)
1 Introduction · 2 Dataset & Data Preparation · 3 DL Fundamentals & Model · 4 System Design & Implementation ·
5 Experimental Methodology · 6 Results · 7 Analysis & Discussion · 8 Real-World Evaluation · 9 Conclusion ·
10 AI Usage Declaration. *(Brief B also asks for a title page, abstract, related work, IEEE/APA references and
an appendix with the team contribution statement. Include them; they don't conflict.)*

### 6.6 Ethics & responsible AI
- **Privacy:** road footage contains faces and plates. Uploaded files are processed in memory/temp and deleted;
  blur faces and plates in any published sample; don't publish raw self-collected footage.
- **Human in the loop:** output is a *review shortlist*, never an automatic fine.
- **Bias:** report performance by condition (day/night, viewpoint, rider vs pillion) and the dataset's country of origin.
- **Licensing:** record each dataset licence; Ultralytics is AGPL-3.0.

---

## 7. Application

`streamlit run app/app.py`
- **Image mode:** upload a photo → annotated boxes, per-class counts, violation warning, inference time, detection table.
- **Video mode:** upload a clip → frame-by-frame detection + **BoT-SORT tracking**, live count of *unique*
  riders per class, FPS, and the final number of unique riders without a helmet.
- Sidebar: weights path, confidence threshold (recall/precision trade-off), violation class, frame stride.
- It uses **our trained `best.pt`** (copy from `runs/<best>/weights/best.pt` to `weights/best.pt`, or set `HELM_WEIGHTS`).

---

## 8. Things needed (resources)

| Category | Needed |
|---|---|
| **People** | 3–5 members (roles in [§10](#10-team-timeline--ai-usage-log)) |
| **Data** | Public helmet dataset (Roboflow Universe / Kaggle), licence recorded; 150–300 self-collected local frames; labelling tool (Roboflow free tier or CVAT) |
| **Compute** | Free GPU: Kaggle Notebooks (T4×2 / P100, ~30 h/week) or Google Colab T4; a laptop for the app (CPU works, GPU better) |
| **Software** | Python 3.10–3.12, PyTorch, `ultralytics`, OpenCV, Streamlit, pandas, matplotlib (`requirements.txt`) |
| **Accounts** | GitHub (this repo), Roboflow, Kaggle and/or Google (Colab, Drive for weights and dataset) |
| **Storage / sharing** | Google Drive or a GitHub Release for the dataset zip and `best.pt` (git-ignored: too large for git) |
| **For user testing** | 3–5 participants, consent note, questionnaire (Google Form) |
| **Deliverables** | Working app (this repo) · technical report PDF · *(confirm)* proposal PDF and demo video < 5 min |

---

## 9. How to run

```bash
git clone https://github.com/<your-account>/Helm-Detector-DL.git && cd Helm-Detector-DL
python -m venv .venv && .venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt                          # GPU: install the CUDA build of torch first (pytorch.org)
python -m pytest                                         # unit tests for data + error-matching logic

# 1. Data: export the dataset in "YOLOv11" format into data/helmet/{train,val,test}/{images,labels}
#    or, for a flat folder of images + .txt labels:
python src/prepare_data.py split --src path/to/raw --dst data/helmet
python src/prepare_data.py stats --root data/helmet --names motorcycle helmet no_helmet
#    -> fix class ids/names in data/helmet.yaml until stats look right

# 2. Train baseline + experiments (one command; or --only B0_baseline E1_scratch)
python src/train.py

# 3. Analysis
python src/compare.py
python src/error_analysis.py --weights runs/B0_baseline/weights/best.pt
python src/robustness.py     --weights runs/B0_baseline/weights/best.pt

# 4. App
mkdir weights && copy runs\B0_baseline\weights\best.pt weights\best.pt
streamlit run app/app.py
```
**On Kaggle/Colab:** clone the repo in a notebook cell, `pip install -r requirements.txt`, upload or attach the
dataset to `data/helmet`, run steps 1–3, then download `runs/*/weights/best.pt` and `results/`.

---

## 10. Team, timeline & AI usage log

### 10.1 Team contribution (fill in)
| Member | Main responsibility | Shared tasks |
|---|---|---|
| _name_ | Data: sourcing, licence, cleaning, labelling own test set, stats | report ch. 2 |
| _name_ | Modelling: training runs, experiments, curves | report ch. 3, 5, 6 |
| _name_ | Evaluation: error analysis, robustness, cost analysis | report ch. 7, 8 |
| _name_ | App + user testing | report ch. 4, 8 |
| _all_ | Report, demo video, AI log | coordination via GitHub issues + PRs |

### 10.2 Timeline (14 weeks)
| Week | Milestone |
|---|---|
| 1–2 | Topic, dataset search, licences, repo setup |
| 3–5 | Data cleaning, class remap, stats, split; collect + label local test set |
| 6 | **Proposal checkpoint** (§1 of this README → 1–2 page PDF) |
| 6–8 | B0 + E1–E3 training (E4–E5 if time allows); curves |
| 9–10 | Error analysis, robustness, model choice |
| 10–11 | App polish (image + video + tracking); user testing |
| 12–13 | Technical report, AI log, demo video |
| 14 | Submit (repo + report + video) |

### 10.3 AI usage log (brief A §20)
| AI tool | Purpose | Project component | How the output was verified |
|---|---|---|---|
| Claude Code (LLM coding assistant) | Read the two project briefs; drafted the repo scaffold, scripts, experiment plan and this README | Whole repo structure, `src/`, `app/`, README | Unit tests (`pytest`, 5 passing on synthetic data); scripts must still be run on the real dataset; architecture facts to be checked against Ultralytics docs and the YOLO11 paper; references to be checked by the team |
| _add rows as you go_ | | | |

---

## 11. Open questions for the lecturer
1. Brief A lists **two** deliverables (app + report). Brief B also had a **Week-6 proposal** and a **< 5 min demo
   video**. Are these still required this semester?
2. Is **Streamlit** acceptable for the app? (Brief B said platforms need instructor approval.)
3. Is user testing with classmates and a security officer enough, or is a real traffic stakeholder required?
4. Report length: brief B said 10–15 pages; brief A gives none.

---

## 12. References
*(Check each one and format in IEEE or APA for the report.)*
1. R. Khanam and M. Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements," arXiv:2410.17725, 2024.
2. G. Jocher and J. Qiu, *Ultralytics YOLO11*, 2024. [Online]. Available: https://github.com/ultralytics/ultralytics
3. N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking," arXiv:2206.14651, 2022.
4. K. Dahiya, D. Singh, and C. K. Mohan, "Automatic detection of bike-riders without helmet using surveillance videos in real-time," in *Proc. IJCNN*, 2016.
5. F. W. Siebert and H. Lin, "Detecting motorcycle helmet use with deep learning," *Accident Analysis & Prevention*, vol. 134, 105319, 2020.
6. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in *Proc. ECCV*, 2014.
7. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proc. CVPR*, 2016.
8. World Health Organization, *Global Status Report on Road Safety 2023*. Geneva: WHO, 2023.
9. Undang-Undang Republik Indonesia No. 22 Tahun 2009 tentang Lalu Lintas dan Angkutan Jalan.
10. U. A. Khan, "Real-Time Helmet Violation Detection using YOLOv11 + BoT-SORT," LinkedIn post (project inspiration).
