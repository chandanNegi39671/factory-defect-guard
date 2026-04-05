<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=2196F3&center=true&vCenter=true&width=700&lines=Factory+Defect+Guard+%F0%9F%8F%AD;YOLOv8+%7C+17+Classes+%7C+29K%2B+Images;mAP%400.5+%3D+83%25+%E2%9C%85" alt="Typing SVG" />
</p>

<p align="center">
  <a href="https://huggingface.co/negi3961/factory-defect-guard">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-factory--defect--guard-yellow?style=for-the-badge" />
  </a>
  <img src="https://img.shields.io/badge/Model-YOLOv8s-00B4D8?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/mAP@0.5-83%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Classes-17-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Images-29%2C354-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
</p>

---

## 🔍 What is this?

> **Factory lines produce thousands of parts per hour. One missed defect = scrapped batch, warranty claims, or worse — a safety incident.**

This project builds an end-to-end automated visual inspection system using **YOLOv8** that detects **17 types of manufacturing defects** across steel surfaces, PCBs, and industrial components — all in a single forward pass, in real time.

No separate model per domain. No manual feature engineering. Just one unified detection pipeline trained on 29,000+ real industrial images.

---

## 📊 Results at a Glance

<table>
  <tr>
    <td align="center"><b>mAP@0.5</b><br><code>83.0%</code></td>
    <td align="center"><b>mAP@0.5:0.95</b><br><code>56.4%</code></td>
    <td align="center"><b>Precision</b><br><code>78.8%</code></td>
    <td align="center"><b>Recall</b><br><code>72.2%</code></td>
    <td align="center"><b>Model Size</b><br><code>22.5 MB</code></td>
    <td align="center"><b>Input</b><br><code>640×640</code></td>
  </tr>
</table>

---

## 🏭 The Problem Space — 17 Defect Classes

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEEL SURFACE (NEU)                          │
│  crazing · inclusion · patches · pitted_surface                 │
│  rolled_in_scale · scratches                                    │
├─────────────────────────────────────────────────────────────────┤
│                    PCB DEFECTS                                  │
│  missing_hole · mouse_bite · open_circuit · short               │
│  spur · spurious_copper                                         │
├─────────────────────────────────────────────────────────────────┤
│              INDUSTRIAL COMPONENTS (MVTec)                      │
│  metal_nut_defect · screw_defect · transistor_defect            │
│  tile_defect · cable_defect                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Pull model from HuggingFace (22.5 MB)
model_path = hf_hub_download(
    repo_id  = "negi3961/factory-defect-guard",
    filename = "best_v6_mc.pt"
)
model = YOLO(model_path)

# Run on any image
results = model.predict("your_part.jpg", conf=0.25)

# Print detections
for box in results[0].boxes:
    print(f"Defect: {model.names[int(box.cls)]}  |  Confidence: {float(box.conf):.2%}")
```

---

## 🗂️ Dataset Pipeline — The Real Work

Most defect detection projects train on a single clean dataset. This project merged **7 different sources** with **6 different annotation formats** into one unified pipeline.

| # | Dataset | Domain | Images | Format |
|---|---------|--------|--------|--------|
| 1 | NEU Surface Defect Database | Steel surface | ~1,800 | Pascal VOC XML |
| 2 | PCB Defects (akhatova) | PCB original | ~1,600 | Pascal VOC XML |
| 3 | PCB Dataset with Augmentations (nakul8820) | PCB augmented | ~2,006 | XML + TXT |
| 4 | PCB Defect Dataset (norbertelter) | PCB YOLO | ~10,668 | YOLO TXT |
| 5 | MVTec AD subset (ipythonx) | Industrial objects | ~428 | Folder structure |
| 6 | Magnetic Tile Defects (alex000kim) | Tile surface | ~2,688 | Image only |
| 7 | Surface Defect (yidazhang07) | Mixed surface | ~4,194 | YOLO TXT |
| | **Total** | | **29,354** | → Unified YOLO |

All Pascal VOC XML annotations were converted to YOLO format programmatically via a custom `voc_to_yolo()` pipeline built in the notebook.

---

## 🧠 Model Architecture & Training

```
Base Model  →  YOLOv8s (Ultralytics)
              ↓
      V5 Training (43 epochs, early stop)
              ↓  mAP: 0.7477
      V6 Fine-tune (60 epochs, full run)
              ↓  mAP: 0.7960
      V6_MC Fine-tune (MC Dropout added)
              ↓  mAP: 0.8300  ← production model
```

**Training Config (V6):**
```yaml
imgsz:     640        optimizer: AdamW
batch:     16         lr0:       0.0001
epochs:    60         mosaic:    1.0
patience:  20         mixup:     0.2
platform:  Kaggle GPU (Tesla T4)
```

**MC Dropout** was injected into the YOLOv8 head's C2f blocks post-training to enable uncertainty estimation — a "Hallucination Shield" that flags low-confidence predictions before they reach downstream systems.

---

## 📈 Per-Class mAP@0.5

| Class | mAP | | Class | mAP |
|-------|-----|---|-------|-----|
| 🟢 `tile_defect` | 99.5% | | 🟡 `rolled_in_scale` | 57.4% |
| 🟢 `pcb_missing_hole` | 99.3% | | 🟡 `screw_defect` | 56.8% |
| 🟢 `pcb_short` | 95.5% | | 🟡 `transistor_defect` | 54.0% |
| 🟢 `patches` | 91.6% | | 🔴 `crazing` | 48.9% |
| 🟢 `pcb_open_circuit` | 90.7% | | | |
| 🟢 `pcb_spurious_copper` | 91.1% | | | |
| 🟢 `inclusion` | 81.3% | | | |
| 🟢 `scratches` | 80.7% | | | |

> `crazing` is the hardest class — it's a subtle surface texture variation that even human inspectors miss under bad lighting. `tile_defect` achieves near-perfect accuracy due to strong visual contrast.

---

## 🔬 Hallucination Shield

A custom uncertainty module built on top of the detector:

```python
# Flags low-confidence predictions before they reach production
shield = HallucinationShield()
result = shield.validate_training_run(V6_METRICS)
# Risk Score: 10/100 — PASS ✅
```

Uses MC Dropout inference (10 forward passes) to compute prediction variance. If variance > threshold, detection is flagged as uncertain rather than passed downstream.

---

## 📁 Repository Structure

```
factory-defect-guard/
├── 📓 notebook124c5.ipynb     ← Full pipeline: data merge → train → eval
├── 📋 Readme.md               ← You are here
├── 📦 Requirement.txt         ← Dependencies
└── 📄 LICENSE                 ← MIT
```

**Model weights** are hosted on HuggingFace (not here — GitHub isn't for 22MB binaries):

🤗 [negi3961/factory-defect-guard](https://huggingface.co/negi3961/factory-defect-guard)

| File | Description | mAP |
|------|-------------|-----|
| `best_v6_mc.pt` | **Recommended** — MC Dropout version | **0.830** |
| `best.pt` | V6 base model | 0.796 |

---

## 🔧 Requirements

```
ultralytics>=8.0.0
huggingface_hub
torch>=2.0.0
Pillow
opencv-python
```

```bash
pip install -r Requirement.txt
```

---

## 🚧 Limitations & Known Issues

| Class | Issue |
|-------|-------|
| `crazing` (48.9%) | Subtle texture — model confuses with normal surface variation |
| `transistor_defect` (54.0%) | Small component, low pixel density at 640px |
| `screw_defect` (56.8%) | High intra-class variation across screw types |

Real factory images may need domain-specific fine-tuning — this model is trained on public benchmarks.

---

## 🗺️ Roadmap

- [ ] Gradio demo on HuggingFace Spaces
- [ ] FastAPI inference endpoint
- [ ] Improve crazing detection (synthetic augmentation)
- [ ] YOLOv8m experiment for +accuracy
- [ ] ONNX export for edge deployment

---

## 👤 Author

**Chandan Singh Ramola**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-chandan--singh--3967ramola-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/chandan-singh-3967ramola)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-negi3961-yellow?style=for-the-badge)](https://huggingface.co/negi3961)
[![GitHub](https://img.shields.io/badge/GitHub-chandanNegi39671-black?style=for-the-badge&logo=github)](https://github.com/chandanNegi39671)

---

<p align="center">
  <i>If this helped you, drop a ⭐ — it helps others find this project.</i>
</p>
