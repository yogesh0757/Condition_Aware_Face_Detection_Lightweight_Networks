# CAFACLite: Condition-Aware Face Anchor Classification for Lightweight Face Detection

[![Paper](https://img.shields.io/badge/ICPR_2026-Paper-blue)](https://github.com/yogesh0757/light_weight_face_detector_lwfd)
[![IJCNN 2025](https://img.shields.io/badge/IJCNN_2025-doi:10.1109/IJCNN64981.2025.11227760-green)](https://ieeexplore.ieee.org/abstract/document/11227760)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.MIT)
[![Python 3.6+](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.1%2B-orange)](https://pytorch.org/)

A PyTorch implementation of **CAFACLite** — a Condition-Aware Face Anchor Classification framework for lightweight face detection under challenging conditions including blur, occlusion, and masked faces.

---

## Overview

Standard lightweight face detectors treat all face anchors uniformly during training, regardless of visual quality. Faces degraded by **blur** or **occlusion** receive the same gradient contribution as clear, well-lit faces, leading to systematic underperformance under these conditions.

**CAFACLite** addresses this through a four-head classification framework that explicitly models face quality conditions using WIDER FACE annotations:

| Head | Purpose |
|---|---|
| `ClassHead` (standard) | Standard binary face/non-face classification |
| `ClassHead_WE` (weighted) | Condition-weighted classification — higher loss weights for blurred/occluded faces |
| `ClassHead_Blur` (blur-specific) | Trained exclusively on blur-labelled face anchors |
| `ClassHead_Occ` (occlusion-specific) | Trained exclusively on occlusion-labelled face anchors |

At inference, all four heads contribute to the final confidence score, enabling the detector to identify faces that the standard classifier would miss.

The framework supports three backbone options:
- **BBLiteV4** — custom lightweight backbone (0.200M params, 0.456 GFLOPs)
- **MobileNetV1 ×0.25** — standard lightweight backbone (0.213M params, 0.545 GFLOPs)
- **ShuffleNetV2 ×0.5** — efficient group-convolution backbone (0.143M params, 0.402 GFLOPs)

---

## Results

### WIDER FACE Validation — CAFACLite (Float32)

| Model | Easy (%) | Medium (%) | Hard (%) | mAP (%) | Blur AP (%) | Occ. AP (%) | Params (M) | GFLOPs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CAFACLite-BB4 (BBLiteV4) | 92.64 | 90.25 | 82.99 | **88.63** | **71.96** | **47.35** | 0.266 | 0.741 |
| CAFACLite-MV1 (MobileNetV1×0.25) | 92.82 | 90.52 | 82.36 | 88.57 | 71.06 | 46.24 | 0.279 | 0.831 |
| CAFACLite-SV2 (ShuffleNetV2×0.5) | 92.40 | 90.33 | 82.38 | 88.37 | 71.52 | 46.60 | 0.430 | 0.828 |

> CAFACLite-BB4 achieves **+5.62% Blur AP** and **+7.60% Occlusion AP** over the EResFD baseline, and **+3.17% / +2.59%** improvements over YOLOv5s while using approximately **26.7× fewer parameters**.

### FDDB Performance (Float32)

| Model | AP (%) |
|:---|:---:|
| CAFACLite-BB4 | 95.89 |
| CAFACLite-MV1 | **95.01** |
| CAFACLite-SV2 | 94.12 |

### MAFA Performance (Float32)

| Model | Whole Set (%) | Mask Set (%) | Unignored Set (%) |
|:---|:---:|:---:|:---:|
| CAFACLite-BB4 | 73.30 | 76.94 | 77.82 |
| CAFACLite-MV1 | **73.50** | **77.57** | 78.07 |
| CAFACLite-SV2 | 72.57 | 75.82 | 77.01 |

---

## Architecture

```
Input Image (W × H × 3)
        │
        ▼
  Backbone (BBLiteV4 / MobileNetV1×0.25 / ShuffleNetV2×0.5)
  ┌─────────────────────────────────────────┐
  │  Stage 1 → C1 (stride 8)               │
  │  Stage 2 → C2 (stride 16)              │
  │  Stage 3 → C3 (stride 32)              │
  └─────────────────────────────────────────┘
        │
        ▼
  Feature Pyramid Network (FPN)
  P1 ← P2 ← P3  (top-down pathway with lateral connections)
        │
        ▼
  SSH Context Modules (SSH1, SSH2, SSH3)
  (3×3 + 5×5 + 7×7 multi-scale context enrichment)
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │            CAFACLite Detection Head             │
  │  ┌──────────────┐  ┌───────────────────────┐   │
  │  │  ClassHead   │  │  ClassHead_WE         │   │
  │  │  (standard)  │  │  (condition-weighted) │   │
  │  ├──────────────┤  ├───────────────────────┤   │
  │  │  ClassHead   │  │  ClassHead_Occ        │   │
  │  │  _Blur       │  │  (occlusion-specific) │   │
  │  └──────────────┘  └───────────────────────┘   │
  │  ┌──────────────┐  ┌───────────────────────┐   │
  │  │  BboxHead    │  │  LandmarkHead         │   │
  │  │  (4 coords)  │  │  (10 coords = 5 pts)  │   │
  │  └──────────────┘  └───────────────────────┘   │
  └────────────────────────────────────────────────┘
        │
        ▼
  NMS → Detected Faces with 5-point Landmarks
```

---

## Repository Structure

```
light_weight_face_detector_lwfd/
│
├── models/
│   ├── cafaclite.py        # CAFACLite detector with 4-head classification
│   ├── net.py              # BBLiteV4, MobileNetV1, FPN, SSH, FeRI modules
│   └── __init__.py
│
├── data/
│   ├── config.py           # Model configs: cfg_CAFACLite, cfg_BV4, cfg_MV1, cfg_SV2
│   ├── wider_face.py       # WIDER FACE dataset loader with blur/occlusion labels
│   ├── data_augment.py     # Training augmentations
│   └── FDDB/
│       └── img_list.txt    # FDDB image list
│
├── layers/
│   ├── modules/
│   │   └── multibox_loss.py   # Multi-task loss with condition-aware supervision
│   └── functions/
│       └── prior_box.py       # Anchor prior box generation
│
├── utils/
│   ├── box_utils.py        # Anchor matching, encode/decode, IoU utilities
│   ├── nms/
│   │   └── py_cpu_nms.py   # CPU Non-Maximum Suppression
│   └── timer.py
│
├── weights/                # Pretrained model weights (download separately)
│   ├── CAFACLite_BV4.pth   # CAFACLite with BBLiteV4 backbone
│   ├── CAFACLite_MV1.pth   # CAFACLite with MobileNetV1×0.25
│   ├── CAFACLite_SV2.pth   # CAFACLite with ShuffleNetV2×0.5
│   ├── WO_CAFACLite_BV4.pth  # Baseline (without CAFAC) — BBLiteV4
│   ├── WO_CAFACLite_MV1.pth  # Baseline (without CAFAC) — MobileNetV1×0.25
│   ├── WO_CAFACLite_SV2.pth  # Baseline (without CAFAC) — ShuffleNetV2×0.5
│   ├── BBLiteV4.pth.tar    # BBLiteV4 backbone pretrained on ImageNet
│   └── mobilenetV1X0.25_pretrain.tar
│
├── train.py                # Training script
├── test_widerface.py       # WIDER FACE evaluation script
├── test_fddb.py            # FDDB evaluation script
├── test_MAFA.py            # MAFA evaluation script
├── detect.py               # Single image detection demo
├── convert_to_onnx.py      # Export model to ONNX format
├── README.md
└── LICENSE.MIT
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yogesh0757/light_weight_face_detector_lwfd.git
cd light_weight_face_detector_lwfd
```

### 2. Install dependencies

```bash
pip install torch torchvision
pip install opencv-python numpy scipy tqdm ptflops
```

> **Requirements:** Python 3.6+, PyTorch 1.1+, torchvision 0.3+, CUDA (optional but recommended for training)

### 3. Download pretrained weights

Download pretrained backbone and detector weights from [Google Drive](https://drive.google.com/drive/folders/1rIEazwm9POodSXE5Sw9H_SDCW4ug71eM?usp=sharing) and place them in the `weights/` directory:

```
weights/
├── CAFACLite_BV4.pth
├── CAFACLite_MV1.pth
├── CAFACLite_SV2.pth
├── WO_CAFACLite_BV4.pth
├── WO_CAFACLite_MV1.pth
├── WO_CAFACLite_SV2.pth
├── BBLiteV4.pth.tar
└── mobilenetV1X0.25_pretrain.tar
```

---

## Data Preparation

### WIDER FACE

```bash
data/widerface/
├── train/
│   ├── images/
│   └── label.txt          # Annotations including blur and occlusion labels
└── val/
    ├── images/
    └── wider_val.txt
```

Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset and annotations (with facial landmark labels) from [Dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0).

### FDDB

```bash
data/FDDB_dataset/
└── images/
    └── [image folders]
```

Download from [Google Drive](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH).

### MAFA

```bash
data/MAFA/
└── images/
    └── [image folders]
```

Download from [IMSG MAFA Dataset](https://imsg.ac.cn/research/maskedface.html).

---

## Training

Before training, review the configuration in `data/config.py`. Key parameters:

```python
cfg_CAFACLite = {
    'batch_size': 8,
    'epoch': 140,
    'decay1': 100,          # LR reduced by 10x at epoch 100
    'decay2': 120,          # LR reduced by 10x at epoch 120
    'image_size': 1024,
    'condition_we_apply': True,   # Enable 4-head CAFAC training
    'out_channel': 32,
}
```

### Train CAFACLite with BBLiteV4 backbone

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --network BBLiteV4
```

### Train CAFACLite with MobileNetV1×0.25 backbone

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --network mobilenet0.25
```

### Train CAFACLite with ShuffleNetV2×0.5 backbone

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --network shufflenet_v2_x0_5
```

### Resume training from a checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --network mobilenet0.25 \
    --resume_net weights/CAFACLite_MV1.pth \
    --resume_epoch 80
```

### Training details

| Setting | Value |
|---|---|
| Optimizer | SGD |
| Initial learning rate | 1×10⁻³ |
| Momentum | 0.9 |
| Weight decay | 5×10⁻⁴ |
| Batch size | 8 |
| Epochs | 140 (decay at 100, 120) |
| GPU | NVIDIA A100 (training) / V100 (SAFAC/CAFACLite) |
| Input image size | 1024×1024 |

---

## Evaluation

### WIDER FACE Validation

#### Step 1: Generate prediction text files

```bash
# CAFACLite with BBLiteV4 backbone
python test_widerface.py \
    --trained_model weights/CAFACLite_BV4.pth \
    --network BBLiteV4 \
    --save_folder ./widerface_evaluate/widerface_txt/

# CAFACLite with MobileNetV1×0.25 backbone
python test_widerface.py \
    --trained_model weights/CAFACLite_MV1.pth \
    --network mobilenet0.25 \
    --save_folder ./widerface_evaluate/widerface_txt/

# CAFACLite with ShuffleNetV2×0.5 backbone
python test_widerface.py \
    --trained_model weights/CAFACLite_SV2.pth \
    --network shufflenet_v2_x0_5 \
    --save_folder ./widerface_evaluate/widerface_txt/
```

Key inference thresholds (from `test_widerface.py`):

| Parameter | Default | Description |
|---|---|---|
| `--confidence_threshold` | 0.02 | Standard head confidence threshold |
| `--confidence_threshold_weight` | 0.05 | Condition-weighted head threshold |
| `--confidence_threshold_blur` | 0.05 | Blur head threshold |
| `--confidence_threshold_occlusion` | 0.09 | Occlusion head threshold |
| `--nms_threshold` | 0.4 | NMS IoU threshold |
| `--keep_top_k` | 750 | Maximum detections per image |

#### Step 2: Compute AP using the WIDER FACE evaluation toolkit

```bash
cd widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```

The evaluation toolkit is adapted from [WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation).

---

### FDDB Evaluation

```bash
# CAFACLite with BBLiteV4 backbone
python test_fddb.py \
    --trained_model weights/CAFACLite_BV4.pth \
    --network BBLiteV4 \
    --dataset /path/to/FDDB_dataset/ \
    --save_folder ./fddb_evaluate/eval/

# CAFACLite with MobileNetV1×0.25 backbone
python test_fddb.py \
    --trained_model weights/CAFACLite_MV1.pth \
    --network mobilenet0.25 \
    --dataset /path/to/FDDB_dataset/ \
    --save_folder ./fddb_evaluate/eval/
```

---

### MAFA Evaluation

```bash
# CAFACLite with BBLiteV4 backbone
python test_MAFA.py \
    --trained_model weights/CAFACLite_BV4.pth \
    --network BBLiteV4 \
    --dataset /path/to/MAFA/ \
    --save_folder ./mafa_evaluate/eval/
```

---

## Single Image Detection

```bash
python detect.py \
    --trained_model weights/CAFACLite_BV4.pth \
    --network BBLiteV4 \
    --confidence_threshold 0.4
```

Detection output is saved as `test.jpg` with bounding boxes drawn in green.

---

## ONNX Export

```bash
python convert_to_onnx.py \
    --trained_model weights/CAFACLite_BV4.pth \
    --network BBLiteV4 \
    --output cafaclite_bv4.onnx
```

---

## Configuration Reference

All model configurations are in `data/config.py`.

### Backbone configurations

```python
cfg_BV4 = {
    'name': 'BBLiteV4',
    'condition_we': [1, 1.4, 1.9],          # per-scale condition weights [P1, P2, P3]
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3}
}

cfg_MV1 = {
    'name': 'mobilenet0.25',
    'condition_we': [1, 1.5, 2.1],
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3}
}

cfg_SV2 = {
    'name': 'shufflenet_v2_x0_5',
    'condition_we': [1, 1.4, 1.9],
    'return_layers': {'stage2': 1, 'stage3': 2, 'conv5': 3}
}
```

### Detector configuration

```python
cfg_CAFACLite = {
    'min_sizes': [[16, 24, 32], [64, 96, 128], [256, 384, 512]],  # anchor sizes per FPN level
    'steps': [8, 16, 32],        # anchor strides
    'variance': [0.1, 0.2],
    'condition_we_apply': True,  # enable/disable CAFAC heads
    'image_size': 1024,
    'out_channel': 32,           # FPN/SSH output channels
}
```

To train **without CAFAC** (standard single-head baseline):
```python
cfg_CAFACLite['condition_we_apply'] = False
```

---

## Comparison with State-of-the-Art Lightweight Detectors

| Model | Hard AP (%) | Blur AP (%) | Occ. AP (%) | Params (M) | GFLOPs |
|:---|:---:|:---:|:---:|:---:|:---:|
| EResFD | 80.42 | 66.34 | 39.75 | **0.09** | **0.30** |
| YOLOv5n0.5 | 73.80 | — | — | 0.45 | 0.57 |
| YOLOv5s | 83.10 | 68.79 | 44.76 | 7.10 | 5.75 |
| **CAFACLite-BB4 (ours)** | **82.99** | **71.96** | **47.35** | 0.266 | 0.741 |
| **CAFACLite-MV1 (ours)** | 82.36 | 71.06 | 46.24 | 0.279 | 0.831 |
| **CAFACLite-SV2 (ours)** | 82.38 | 71.52 | 46.60 | 0.430 | 0.828 |

---

## Citation

If you use this code or the CAFACLite/CAFAC framework in your research, please cite:

```bibtex
@inproceedings{aggarwal2026cafaclite,
  title     = {Condition-Aware Face Anchor Classification for Lightweight Face Detection},
  author    = {Aggarwal, Yogesh and Guha, Prithwijit},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```

For the BBLite backbone series (IJCNN 2025), please also cite:

```bibtex
@inproceedings{aggarwal2025bblite,
  author    = {Aggarwal, Yogesh and Guha, Prithwijit},
  booktitle = {2025 International Joint Conference on Neural Networks (IJCNN)},
  title     = {Designing Customized Lightweight Backbones for the Face Detection Task},
  year      = {2025},
  pages     = {1--10},
  doi       = {10.1109/IJCNN64981.2025.11227760}
}
```

---

## Acknowledgements

This work builds upon:
- [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) — the base detection framework
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) — PyTorch RetinaFace implementation
- [WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation) — WIDER FACE evaluation toolkit
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) — anchor generation utilities

---

## License

This project is released under the [MIT License](LICENSE.MIT).
