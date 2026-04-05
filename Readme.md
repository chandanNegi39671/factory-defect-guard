# 🏭 Factory Defect Guard — Multi-Domain Industrial Defect Detection

> Automated quality inspection for manufacturing lines using YOLOv8 trained on 29,000+ images across steel, PCB, and industrial component domains.

![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-blue)
![Classes](https://img.shields.io/badge/Classes-17-green)
![mAP](https://img.shields.io/badge/mAP@0.5-79.6%25-orange)
![Dataset](https://img.shields.io/badge/Images-29%2C354-purple)

---

## Problem Statement

Manual visual inspection on factory floors is slow, inconsistent, and expensive. This project builds an automated defect detection pipeline that identifies 17 types of manufacturing defects across steel surfaces, PCBs, and industrial components in real time.

---

## Datasets Used

| Dataset | Domain | Source |
|---|---|---|
| NEU Surface Defect Database | Steel surface defects | Kaggle (kaustubhdikshit) |
| PCB Defect Dataset (Original) | Printed circuit boards | Kaggle (akhatova) |
| PCB Dataset with Augmentations | PCBs augmented | Kaggle (nakul8820) |
| PCB Defect Dataset (Norbert) | PCBs YOLO format | Kaggle (norbertelter) |
| MVTec AD Subset | Industrial objects | Kaggle (ipythonx) |
| Magnetic Tile Defects | Tile surface defects | Kaggle (alex000kim) |
| Surface Defect (YidaZhang) | Mixed surface | Kaggle (yidazhang07) |

All datasets were unified into a single YOLO-format pipeline with standardized 17-class annotations.

---

## ML Approach

- **Model:** YOLOv8s (small variant, balance of speed and accuracy)
- **Framework:** Ultralytics
- **Training Platform:** Kaggle (GPU T4 x2)
- **Annotation formats handled:** Pascal VOC XML → YOLO TXT conversion
- **Augmentation:** Mosaic (1.0), MixUp (0.2)
- **Optimizer:** AdamW, lr0 = 0.0001
- **Multiple training runs:** V5 → V6 (fine-tuned from previous checkpoint)

---

## 17 Defect Classes

**Steel Surface (NEU):** `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled_in_scale`, `scratches`

**PCB Defects:** `pcb_missing_hole`, `pcb_mouse_bite`, `pcb_open_circuit`, `pcb_short`, `pcb_spur`, `pcb_spurious_copper`

**Industrial Components (MVTec-derived):** `metal_nut_defect`, `screw_defect`, `transistor_defect`, `tile_defect`, `cable_defect`

---

## Training Configuration (V6 — Final)
```yaml
epochs: 60
imgsz: 640
batch: 16
optimizer: AdamW
lr0: 0.0001
mosaic: 1.0
mixup: 0.2
patience: 20
cls_loss_weight: 2.0
box_loss_weight: 9.0
```

---

## Results

| Metric | Value |
|---|---|
| mAP@0.5 (overall) | **79.6%** |
| mAP@0.5:0.95 | 56.4% |
| Precision | 78.8% |
| Recall | 72.2% |
| Epochs trained | 60 |

**Class-wise mAP@0.5 highlights:**
- `tile_defect`: 99.5%
- `pcb_missing_hole`: 99.3%
- `pcb_short`: 95.5%
- `patches`: 91.6%
- `crazing`: 48.9% *(hardest class — subtle surface texture)*

---

## Model

Trained weights are available on Hugging Face:

🤗 **[negi3961/factory-defect-guard](https://huggingface.co/negi3961/factory-defect-guard)**
```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id='negi3961/factory-defect-guard', filename='best.pt')
model = YOLO(path)
results = model.predict('your_image.jpg')
```

---

## Repo Structure
