# VK-BradyNet

**VK-BradyNet (Visual–Kinematic Bradykinesia Network)** is a dual-modal framework for video-based assessment of Parkinsonian hand bradykinesia across three MDS-UPDRS Part III hand tasks:

- **Finger Tapping (FT; item 3.4)**
- **Hand Movements (HM; item 3.5)**
- **Pronation–Supination Movements of Hands (item 3.6; referred to hereafter as Forearm Rotation [FR])**

The framework combines an RGB spatiotemporal representation learned by **MViTv2-S** with a structured hand-motion representation derived from **21 MediaPipe hand landmarks**. The keypoint branch uses motion- and validity-aware temporal weighting, and the two modalities are integrated through adaptive gated fusion.

This repository contains the project code used for video preprocessing, hand ROI extraction, RGB–keypoint representation construction, VK-BradyNet training and evaluation, attribution visualization, and reproduction of selected published comparison methods.

---

## 1. Study overview

The study included **227 patients with Parkinson's disease** and **5,364 paired RGB–keypoint video clips** with MDS-UPDRS severity scores restricted to **0–2**.

The paired motor-severity dataset contained:

| Task | Total clips |
|---|---:|
| Hand Movements (HM) | 1,950 |
| Finger Tapping (FT) | 1,500 |
| Forearm Rotation (FR) | 1,914 |
| **Total** | **5,364** |

The data were partitioned at the **patient level** into training, validation, and test subsets at an approximate 8:1:1 ratio:

| Split | Clips |
|---|---:|
| Training | 4,290 |
| Validation | 537 |
| Test | 537 |

All task-specific recordings, left- and right-hand performances, and derived clips from the same patient were assigned exclusively to one subset.

A separate hand-detection dataset contained **8,806 manually annotated frames**:

| Task | Images |
|---|---:|
| HM | 2,910 |
| FT | 2,933 |
| FR | 2,963 |
| **Total** | **8,806** |

---

## 2. Repository components

The current repository is organized around four major components:

```text
1. Data preprocessing
2. Hand detection and ROI extraction
3. VK-BradyNet
4. Published-method comparison projects
```

The corresponding code includes utilities for:

- FFmpeg-based video standardization and temporal segmentation;
- frame extraction;
- YOLO11n hand localization;
- ROI video reconstruction;
- MediaPipe Hands landmark extraction;
- 10-dimensional keypoint feature construction;
- RGB–keypoint pairing;
- MViTv2-S-based RGB representation learning;
- speed-aware keypoint temporal modeling;
- adaptive gated fusion;
- three-class MDS-UPDRS severity classification;
- evaluation and qualitative attribution visualization;
- reproduction of FastEval Parkinsonism, Islam et al. (2023), and VisionMD (2025).

---

## 3. End-to-end workflow

```text
Raw MTS / MP4 clinical video
            |
            v
FFmpeg standardization
H.264 / YUV 4:2:0 / 25 fps / CRF 16
audio removed / slow preset
            |
            v
Consecutive non-overlapping 2-s clips
(50 frames per complete clip)
            |
            v
Frame extraction to PNG
            |
            v
Hand-detection dataset construction
            |
            v
YOLO11n hand localization
            |
            v
Hand ROI extraction and ROI-video reconstruction
            |
            +-------------------------------+
            |                               |
            v                               v
      RGB ROI video                  MediaPipe Hands
            |                               |
            v                               v
        MViTv2-S                    21 hand landmarks
            |                               |
            |                               v
            |                      10D keypoint features
            |                               |
            |                               v
            |                         [T, 21, 10]
            |                               |
            +---------------+---------------+
                            |
                            v
                 Paired multimodal sample
                            |
                            v
                       VK-BradyNet
                            |
                            v
                  MDS-UPDRS score 0 / 1 / 2
```

---

# Part I. Video preprocessing

## 4. Video standardization and temporal segmentation

The raw recordings are standardized with **FFmpeg** before downstream analysis.

The preprocessing procedure uses:

```text
Container          MP4
Video codec        H.264
Pixel format       YUV 4:2:0
Frame rate         25 fps
CRF                16
Encoding preset    slow
Audio              removed
```

After standardization, each task-specific recording is divided into **consecutive non-overlapping 2-s clips**. At 25 fps, each complete clip contains **50 frames**.

Clips overlapping the transition between left- and right-hand performances are excluded so that each retained clip corresponds to one task side and one task- and side-specific clinical score.

### Current preprocessing scripts

```text
MTS mp4 cilp.py
mp4-png.py
clip mp4.py
```

### `MTS mp4 cilp.py`

Performs FFmpeg-based format standardization and segmentation of MTS/MP4 recordings into short MP4 clips.

### `mp4-png.py`

Decodes MP4 clips frame by frame and saves the complete frame sequence as PNG images while preserving spatial resolution and temporal order.

### `clip mp4.py`

Reconstructs MP4 video from ordered frame folders when frame-based intermediate processing is required.

---

# Part II. Hand detection and ROI extraction

## 5. Hand-detection dataset

Frames sampled from the extracted image sequences are manually annotated for the **task-performing hand**.

Only the performing hand is treated as the detection target. When the contralateral hand is visible, it is not annotated as the target.

The hand-detection dataset uses the same patient-level separation principle as the motor-severity dataset.

---

## 6. YOLO11n hand localization

Hand localization is implemented with **YOLO11n** using the Ultralytics framework.

Four detector experiments are used in the study:

```text
HM-specific detector
FT-specific detector
FR-specific detector
combined HM–FT–FR detector
```

The reported training configuration is:

| Parameter | Setting |
|---|---:|
| Input size | 640 × 640 |
| Maximum epochs | 150 |
| Batch size | 4 |
| Optimizer | SGD |
| Model selection | validation performance |
| Checkpoint | best validation checkpoint |

The checkpoint from the **combined HM–FT–FR detector** is used for frame-wise hand localization and ROI extraction in all standardized video clips.

### Current ROI package files

```text
train.py
detect.py
data.yml
best.pt
yolo11n.pt
```

---

## 7. ROI extraction

Each standardized video clip is processed frame by frame using the trained YOLO11n detector.

Current ROI rule:

```text
confidence threshold = 0.25
```

When multiple detections are produced in the same frame, the bounding box with the **highest confidence score** is selected.

For each clip:

1. valid bounding boxes are collected across frames;
2. the maximum detected width and height define a clip-specific ROI canvas;
3. each detected hand region is cropped without geometric resizing;
4. the crop is centered on the fixed canvas;
5. the remaining area is zero-padded;
6. if no valid hand is detected in a frame, a black frame of the same canvas size is inserted;
7. ROI frames are re-encoded in their original temporal order.

This preserves temporal alignment while reducing task-irrelevant image content.

---

# Part III. Hand-keypoint representation

## 8. MediaPipe Hands

ROI videos are processed using **MediaPipe Hands** in video-tracking mode.

Current settings:

```text
static_image_mode           = False
max_num_hands               = 1
min_detection_confidence    = 0.5
min_tracking_confidence     = 0.5
```

Each frame is represented by 21 anatomical hand landmarks:

```text
P ∈ R^(T × 21 × 3)
```

where each landmark contains `(x, y, z)` coordinates.

If no valid hand is detected, the corresponding landmark coordinates are initialized to zero and a frame-level validity indicator is recorded.

### Current scripts

```text
only npy.py
npy kvideo.py
```

`only npy.py` generates the keypoint feature arrays.

`npy kvideo.py` additionally produces a landmark-visualization video for inspection.

---

## 9. Landmark normalization and 10D features

The landmark coordinates are normalized using:

- landmark 0 (wrist) as the spatial origin;
- the wrist-to-middle-finger MCP distance (landmarks 0 and 9) as the scale factor.

For each landmark, the representation contains:

```text
1. normalized x
2. normalized y
3. normalized z
4. frame-to-frame displacement in x
5. frame-to-frame displacement in y
6. frame-to-frame displacement in z
7. motion magnitude
8. temporal variation in motion magnitude
9. wrist-relative distance
10. keypoint-detection validity
```

The resulting keypoint tensor is:

```text
[T, 21, 10]
```

The frame-level validity value is propagated across all 21 landmarks for the corresponding frame.

---

## 10. RGB–keypoint pairing

Each multimodal sample contains:

```text
ROI video clip
keypoint feature array (.npy)
MDS-UPDRS severity label
```

The RGB clip and keypoint tensor are paired using a unique sample identifier so that they correspond to the same temporal sequence.

The current manifest utility is:

```text
my_csv.py
```

with rows of the form:

```text
video_path keypoint_npy_path label
```

The utility generates:

```text
train.csv
val.csv
test.csv
```

---

# Part IV. VK-BradyNet

## 11. Model architecture

VK-BradyNet contains four principal components:

```text
1. MViTv2-S RGB branch
2. speed-aware hand-keypoint temporal branch
3. adaptive gated multimodal fusion
4. three-class severity classification head
```

The same architecture is trained, validated, and tested independently for HM, FT, and FR.

---

## 12. RGB branch

The visual backbone is **MViTv2-S**, pretrained on **Kinetics-400**.

For each ROI clip:

```text
input tensor = [B, 3, 16, 224, 224]
```

The MViTv2-S hierarchy contains 16 multiscale Transformer blocks distributed across four stages:

```text
1 / 2 / 11 / 2 blocks
```

The visual representation has dimension:

```text
768
```

The Kinetics-400-pretrained RGB backbone remains trainable during task-specific fine-tuning.

### Main pretrained checkpoint

```text
MViTv2_S_16x4_k400.pyth
```

---

## 13. Keypoint temporal branch

The keypoint input is:

```text
K ∈ R^(B × T × 21 × 10), T = 16
```

If the original sequence contains more than 16 frames, **16 temporal positions are uniformly selected**. If fewer than 16 frames are available, the final frame is repeated until the target length is reached.

At each time step:

```text
21 landmarks × 10 features
        |
        v
210-dimensional vector
        |
        v
128-dimensional embedding
```

Embedding dropout:

```text
0.3
```

The temporal encoder is a **two-layer Transformer encoder** with:

| Parameter | Setting |
|---|---:|
| Embedding dimension | 128 |
| Self-attention heads | 4 |
| Feed-forward dimension | 512 |
| Activation | GELU |
| Dropout | 0.3 |

The main implementation is located in:

```text
SlowFast-main/slowfast/models/video_model_builder.py
```

---

## 14. Speed-aware temporal weighting

The frame-wise mean motion magnitude is computed across the 21 landmarks:

```text
mean motion magnitude
```

and combined with the frame-level keypoint-validity term.

The temporal relevance score follows:

```text
r_t = c_t [ sigmoid(-alpha * mean_motion_t) + beta ]
```

with:

```text
alpha = 10
beta  = 0.05
eps   = 1e-6
```

The normalized temporal weight is scaled so that the total weight is approximately equal to the sequence length `T`.

The same motion-informed temporal weights are used:

```text
1. before Transformer encoding
2. during sequence-level temporal pooling
```

Low-motion temporal segments therefore receive greater relative emphasis, while invalid keypoint frames are suppressed through the validity term.

---

## 15. Adaptive gated multimodal fusion

The RGB and keypoint representations are first concatenated:

```text
RGB feature       = 768D
Keypoint feature  = 128D
Combined feature  = 896D
```

A 128-dimensional sigmoid gate is generated from the joint representation and applied element-wise to the keypoint feature.

The fusion scale is:

```text
lambda = 1.0
```

The final fused representation is:

```text
[RGB feature ; gated keypoint feature]
```

with dimension:

```text
896
```

The RGB stream is retained directly, while the keypoint representation is adaptively modulated.

---

## 16. Classification head

The fused feature is passed through:

```text
896D fused representation
        |
        v
Dropout(0.6)
        |
        v
Linear: 896 -> 768
        |
        v
GELU
        |
        v
Dropout(0.7)
        |
        v
Linear: 768 -> 3
        |
        v
Scores 0 / 1 / 2
```

Training uses standard cross-entropy loss.

---

## 17. Training configuration

The reported VK-BradyNet training configuration is:

| Parameter | Setting |
|---|---:|
| Optimizer | AdamW |
| Base learning rate | 3 × 10^-6 |
| Weight decay | 0.1 |
| LR schedule | cosine |
| Warm-up epochs | 3 |
| Warm-up learning rate | 3 × 10^-7 |
| Final cosine learning rate | 3 × 10^-7 |
| Training epochs | **50** |
| Batch size | 2 |
| Gradient clipping | max L2 norm = 1.0 |
| Mixed-precision training | enabled |

The RGB backbone, keypoint encoder, gating module, and classification head are optimized jointly.

### Main project files

```text
MVITv2_S_PD_3cls_rgb_keypoint_fusion.yaml
MVITv2_S_PD_3cls_rgb_only.yaml
MVITv2_S_PD_3cls_keypoint_only.yaml
train_rgb_keypoint_fusion.py
train_keypoint_only.py
train_net_validation_best.py
test_rgb_keypoint_fusion_metrics.py
```

Framework-level files are under:

```text
SlowFast-main/
```

---

## 18. Ablation variants

The validation-set ablation study uses five variants:

| Variant | RGB | Keypoint | Gate | Speed-aware weighting |
|---|:---:|:---:|:---:|:---:|
| RGB-only | ✓ | – | – | – |
| Keypoint-only | – | ✓ | – | – |
| Direct fusion | ✓ | ✓ | – | – |
| Gated fusion | ✓ | ✓ | ✓ | – |
| Proposed | ✓ | ✓ | ✓ | ✓ |

Validation results reported in the paper are:

| Variant | Task | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|
| RGB-only | HM | 68.21 | 67.01 |
|  | FT | 61.33 | 60.84 |
|  | FR | 70.83 | 70.22 |
| Keypoint-only | HM | 66.67 | 65.80 |
|  | FT | 60.67 | 59.98 |
|  | FR | 69.79 | 69.02 |
| Direct fusion | HM | 75.90 | 75.64 |
|  | FT | 68.00 | 67.54 |
|  | FR | 77.60 | 77.13 |
| Gated fusion | HM | 76.41 | 75.42 |
|  | FT | 69.33 | 68.91 |
|  | FR | 78.65 | 78.08 |
| **Proposed** | **HM** | **78.46** | **78.27** |
|  | **FT** | **72.00** | **71.49** |
|  | **FR** | **79.69** | **79.17** |

---

# Part V. Evaluation

## 19. Evaluation protocol

Final motor-severity performance is assessed independently on the held-out test subsets for HM, FT, and FR.

Model selection and ablation experiments use the validation subsets; the test subsets are not used for hyperparameter optimization.

The evaluation includes:

```text
Accuracy
class-wise Precision
class-wise Recall
class-wise F1
Macro-F1
Quadratic Weighted Kappa (QWK)
one-vs-rest ROC/AUC
macro-average AUC
micro-average AUC
reliability diagrams
```

For uncertainty in accuracy, the study uses **patient-level cluster bootstrap resampling**:

```text
bootstrap replicates = 2000
independent resampling unit = patient
95% CI = percentile interval [2.5%, 97.5%]
```

All clips belonging to a sampled patient are retained together within each bootstrap replicate.

---

## 20. Internal test performance

The reported held-out test results are:

| Task | Accuracy (%) | Macro-F1 (%) | 95% CI for accuracy (%) | QWK |
|---|---:|---:|---:|---:|
| HM | 76.41 | 74.80 | 71.28–81.54 | 0.789 |
| FT | 69.33 | 67.82 | 62.67–76.00 | 0.759 |
| FR | 77.60 | 77.48 | 71.86–83.33 | 0.761 |

Class-wise results:

| Task | Severity | Precision (%) | Recall (%) | F1-score (%) |
|---|---|---:|---:|---:|
| HM | Score 0 | 69.23 | 96.92 | 80.77 |
| HM | Score 1 | 83.33 | 46.15 | 59.41 |
| HM | Score 2 | 82.35 | 86.15 | 84.21 |
| FT | Score 0 | 80.39 | 80.39 | 80.39 |
| FT | Score 1 | 52.63 | 43.48 | 47.62 |
| FT | Score 2 | 70.49 | 81.13 | 75.44 |
| FR | Score 0 | 79.17 | 89.06 | 83.82 |
| FR | Score 1 | 70.77 | 71.88 | 71.32 |
| FR | Score 2 | 83.64 | 71.88 | 77.31 |

---

## 21. Evaluation and visualization scripts

Current evaluation/visualization files include:

```text
test_rgb_keypoint_fusion_metrics.py
GRAD.py
Video Grad-CAM Visualization.py
Video_GradCAM_Visualization_final_optimized.py
```

The evaluation code produces classification and discrimination outputs such as:

```text
predictions
confusion matrices
class-wise metrics
ROC curves
precision–recall curves
metric summaries
```

The qualitative visualization workflow is used to display:

```text
original clinical frame with detected hand
extracted hand ROI
MediaPipe landmarks
landmark-level attribution
RGB visual attribution
RGB attribution overlay
```

These visualizations are intended as qualitative representations of the learned RGB and keypoint information.

---

# Part VI. External validation and reproduced published methods

## 22. Zero-shot external validation

The paper reports zero-shot external validation on **HUBU-FIS** for FT.

No retraining or fine-tuning of VK-BradyNet is performed for this external evaluation.

From the available external cohort, **123 FT videos from 66 patients with Parkinson's disease** with severity scores 0–2 are retained:

| Severity | Videos |
|---|---:|
| Score 0 | 13 |
| Score 1 | 72 |
| Score 2 | 38 |

External performance:

| Dataset | Task | Accuracy (%) | Macro-F1 (%) | QWK |
|---|---|---:|---:|---:|
| HUBU-FIS | FT | 59.35 | 58.50 | 0.567 |

---

## 23. Published-method comparison projects

Three public methods/analysis pipelines are reproduced and adapted to the common Score 0–2 setting:

```text
FastEval Parkinsonism
Islam et al. (2023)
VisionMD (2025)
```

The reproduced methods are evaluated using the corresponding task-specific partitions of the study dataset.

These implementations are organized separately from VK-BradyNet.

A practical directory layout is:

```text
comparison_methods/
├── FastEval_150_Test_PyCharm/
├── Islam2023_FT_PyCharm/
└── VisionMD_HM_PyCharm/
```

---

## 24. FastEval Parkinsonism

Upstream method:

```text
FastEval Parkinsonism
Yang et al.
npj Digital Medicine, 2024
```

Public repository:

```text
https://github.com/yuyuan871111/fast_eval_Parkinsonism
```

For the study comparison, FastEval is reproduced using its public implementation and adapted to the common three-class Score 0–2 setting.

The reproduced pipeline retains its 3D hand-keypoint sequence processing, normalization, missing-value handling, augmentation, and temporal modeling.

Study comparison setting:

```text
Training data   = our FT training cohort
Test data       = our held-out FT test set
Task            = FT
Evaluation      = 150 clips
```

Reported performance:

| Accuracy (%) | Macro-F1 (%) | QWK |
|---:|---:|---:|
| 62.67 | 62.40 | 0.621 |

Project scripts include setup, test-set checking, inference/evaluation utilities, and comparison-table output generation.

---

## 25. Islam et al. (2023)

Upstream study:

```text
Islam MS et al.
Using AI to measure Parkinson's disease severity at home.
npj Digital Medicine. 2023.
```

Public repository:

```text
https://github.com/ROC-HCI/finger-tapping-severity
```

The reproduced method uses clinically informed MediaPipe-derived kinematic descriptors representing aspects of:

```text
movement amplitude
speed
periodicity
rhythm
interruptions
decrement
wrist motion
```

followed by LightGBM-based prediction.

Because the original method was formulated for continuous severity estimation, predicted values are mapped to the nearest severity class within the 0–2 range for the present comparison.

Study comparison setting:

```text
Training data   = our FT training cohort
Test data       = our held-out FT test set
Task            = FT
Evaluation      = 150 clips
```

Reported performance:

| Accuracy (%) | Macro-F1 (%) | QWK |
|---:|---:|---:|
| 56.67 | 56.37 | 0.549 |

---

## 26. VisionMD (2025)

Upstream study:

```text
Acevedo G, Lange F, Calonge C, et al.
VisionMD: an open-source tool for video-based analysis of motor function
in movement disorders.
npj Parkinson's Disease. 2025.
```

Public repository:

```text
https://github.com/mea-lab/VisionMD-DesktopApp
```

VisionMD is reproduced using its public kinematic-analysis pipeline.

Hand landmarks are converted into task-specific motion signals and processed to derive features describing:

```text
amplitude
speed
timing
variability
pauses
hesitations
```

Because VisionMD was developed primarily as a kinematic-analysis framework rather than a three-class severity classifier, a **LightGBM classifier** is trained using the HM training cohort, with validation-based model selection and final evaluation on the same held-out HM test set used for VK-BradyNet.

Study comparison setting:

```text
Training data   = our HM training cohort
Test data       = our held-out HM test set
Task            = HM
Evaluation      = 195 clips
```

Reported performance:

| Accuracy (%) | Macro-F1 (%) | QWK |
|---:|---:|---:|
| 70.77 | 70.43 | 0.752 |

---

## 27. Comparison summary

| Evaluation setting | Method | Task | Test data | Accuracy (%) | Macro-F1 (%) | QWK |
|---|---|---|---|---:|---:|---:|
| Zero-shot external validation | VK-BradyNet | FT | HUBU-FIS, 123 videos | 59.35 | 58.50 | 0.567 |
| Published-method reproduction | FastEval Parkinsonism | FT | held-out FT test set, 150 clips | 62.67 | 62.40 | 0.621 |
| Published-method reproduction | Islam et al. (2023) | FT | held-out FT test set, 150 clips | 56.67 | 56.37 | 0.549 |
| Internal test | VK-BradyNet | FT | held-out FT test set, 150 clips | 69.33 | 67.82 | 0.759 |
| Published-tool reproduction | VisionMD (2025) | HM | held-out HM test set, 195 clips | 70.77 | 70.43 | 0.752 |
| Internal test | VK-BradyNet | HM | held-out HM test set, 195 clips | 76.41 | 74.80 | 0.789 |

The reproduced published methods were originally developed under different objectives, datasets, and evaluation protocols. Their implementations here are adapted to the common Score 0–2 setting for comparison under the study protocol.

---

# Part VII. Dependencies and execution

## 28. Core dependencies

Across the current projects, the main software dependencies include:

```text
Python
PyTorch
torchvision
torchaudio
PySlowFast
PyAV
OpenCV
NumPy
pandas
scikit-learn
matplotlib
fvcore
iopath
pytorchvideo
simplejson
tensorboard
psutil
MediaPipe
Ultralytics
LightGBM
FFmpeg
FFprobe
```

The comparison projects include their own dependency/setup files for the corresponding public implementations.

---

## 29. Recommended execution order

A practical order for the main VK-BradyNet workflow is:

```text
1. Standardize raw video and generate 2-s clips
      MTS mp4 cilp.py

2. Extract frame images
      mp4-png.py

3. Prepare the hand-detection dataset and YOLO-format annotations

4. Train YOLO11n hand-localization models
      train.py

5. Generate hand-centered ROI videos
      detect.py

6. Extract MediaPipe-based 10D keypoint features
      only npy.py

7. Optionally inspect landmark tracking
      npy kvideo.py

8. Pair ROI videos and keypoint arrays
      my_csv.py

9. Train VK-BradyNet independently for HM, FT, and FR
      train_rgb_keypoint_fusion.py

10. Evaluate held-out test sets
      test_rgb_keypoint_fusion_metrics.py

11. Generate qualitative attribution visualizations
      Video_GradCAM_Visualization_final_optimized.py

12. Run the published-method comparison projects separately
      FastEval Parkinsonism
      Islam et al. (2023)
      VisionMD (2025)
```

---

## 30. Data availability and privacy

The clinical videos are not distributed with this repository because they contain sensitive patient information and are subject to ethical and privacy restrictions.

Users should configure their own data paths locally and handle clinical data in accordance with applicable institutional, ethical, and privacy requirements.

---

## 31. Third-party software

The project builds on or interfaces with several open-source projects, including:

```text
PySlowFast
Ultralytics YOLO11
MediaPipe Hands
FastEval Parkinsonism
finger-tapping-severity
VisionMD
```

Please consult the corresponding upstream repositories for their licenses, installation instructions, pretrained resources, and original documentation.

---

## 32. Notes

This README is aligned with the methods, experimental settings, and reported results of the accompanying VK-BradyNet manuscript.

Where project scripts contain additional development options or historical configuration artifacts, the manuscript-reported experimental protocol is treated as the reference description for this repository-level documentation.
