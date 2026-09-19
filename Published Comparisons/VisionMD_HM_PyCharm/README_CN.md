# VisionMD Hand Movement reproduction for our HM cohort
## PyCharm-ready project

This project adapts the public VisionMD implementation to our MDS-UPDRS 3.5
Hand Movements (HM) dataset.

Paper:
Acevedo G, Lange F, Calonge C, et al.
VisionMD: an open-source tool for video-based analysis of motor function
in movement disorders. npj Parkinson's Disease. 2025;11:27.

Official source:
https://github.com/mea-lab/VisionMD-DesktopApp

Important methodological point:
VisionMD is primarily a kinematic-analysis tool, not a published 0/1/2
severity classifier. Therefore this project keeps the VisionMD HM
representation and feature extraction, then adds a fixed downstream
LightGBM classifier only so that the reproduced method can be evaluated
with the same exact Accuracy used for VK-BradyNet.

The resulting comparison should be described as:

    VisionMD kinematic features + LightGBM

rather than implying that the original VisionMD paper published this
three-class classifier.

------------------------------------------------------------
1. WHAT IS RETAINED FROM VISIONMD
------------------------------------------------------------

The public VisionMD implementation uses MediaPipe hand landmarks and a
task-specific HM motion signal.

For HM, the current public source code computes the raw signal as the
average wrist-to-fingertip distance for the index, middle, and ring
fingers:

    s(t) = [d(index_tip,wrist)
          + d(middle_tip,wrist)
          + d(ring_tip,wrist)] / 3

The public code then normalizes the signal by a hand-size factor and uses
PeakfinderSignalAnalyzer to:
- normalize and resample the signal to 60 Hz;
- detect opening/closing cycles;
- calculate 23 kinematic measures including amplitude, speed,
  RMS velocity, opening/closing speed, cycle duration, variability,
  frequency, amplitude decay, velocity decay, pauses, and hesitations.

This project imports the official VisionMD PeakfinderSignalAnalyzer
directly from the cloned repository. The signal analyzer is not rewritten.

NOTE:
The 2025 paper text describes the HM signal more simply as the
middle-fingertip-to-wrist distance normalized by hand length. The current
public VisionMD source code instead averages index-, middle-, and
ring-fingertip distances to the wrist. Because this project is intended to
reproduce the publicly released implementation, the current public source
code definition is used and the repository commit is recorded.

------------------------------------------------------------
2. OUR HM DATA LAYOUT
------------------------------------------------------------

Place videos as:

data/
  train/
    0/*.mp4
    1/*.mp4
    2/*.mp4
  val/
    0/*.mp4
    1/*.mp4
    2/*.mp4
  test/
    0/*.mp4
    1/*.mp4
    2/*.mp4

Expected numbers for the current study:
- train: 1560 HM clips
- validation: 195 HM clips
- test: 195 HM clips

The scripts issue warnings if the counts differ; they do not fabricate or
discard data.

If a ZIP contains class folders 0/1/2 anywhere in its path:

    python scripts/00_import_split.py --zip "D:\HM_train.zip" --split train
    python scripts/00_import_split.py --zip "D:\HM_val.zip"   --split val
    python scripts/00_import_split.py --zip "D:\HM_test.zip"  --split test

------------------------------------------------------------
3. PYCHARM / PYTHON ENVIRONMENT
------------------------------------------------------------

Recommended:
- Python 3.10
- PyCharm
- Windows 10/11
- CPU is sufficient for feature extraction and LightGBM
- GPU is not required

Install:

    pip install -r requirements.txt

Then download the official VisionMD repository:

    python scripts/01_setup_official_visionmd.py

Download the MediaPipe Hand Landmarker model:

    python scripts/02_download_hand_model.py

------------------------------------------------------------
4. BUILD MANIFESTS
------------------------------------------------------------

Run:

    python scripts/03_build_manifest.py --split train
    python scripts/03_build_manifest.py --split val
    python scripts/03_build_manifest.py --split test

Manifest fields:
- filename
- relative_path
- label
- patient_id
- side
- roi_x, roi_y, roi_w, roi_h

Patient ID is parsed from the filename prefix before the first "_".

Side can be:
- left
- right
- auto

Default is `auto`.

If real left/right metadata are available, replace `auto` with the true
side. Real metadata are preferable to automatic handedness selection.

ROI columns are optional. If left empty, the whole frame is analyzed.
If the clips are already hand-centered/cropped, leave them empty.

------------------------------------------------------------
5. EXTRACT VISIONMD HM FEATURES
------------------------------------------------------------

Run:

    python scripts/04_extract_visionmd_hm_features.py --split train
    python scripts/04_extract_visionmd_hm_features.py --split val
    python scripts/04_extract_visionmd_hm_features.py --split test

Output:

    features/train_features.csv
    features/val_features.csv
    features/test_features.csv

Failures:

    results/extraction_failures_train.csv
    results/extraction_failures_val.csv
    results/extraction_failures_test.csv

The current public VisionMD HM code rejects a video when the requested
hand is missing in more than 10% of frames. This project applies the same
10% threshold and linearly interpolates the remaining missing landmarks.

Do NOT silently report an Accuracy based only on successfully extracted
test clips if some held-out clips fail. The number of successful test clips
must be reported.

------------------------------------------------------------
6. TRAIN THE DOWNSTREAM CLASSIFIER
------------------------------------------------------------

Run:

    python scripts/05_train_evaluate.py --mode val_tune

The LightGBM classifier is trained on the VisionMD kinematic features.

`val_tune`:
- a small predefined parameter grid is evaluated on the validation set;
- Macro-F1 selects the model;
- validation Accuracy breaks ties;
- the held-out test set is not used for parameter selection;
- the selected configuration is finally fit on train + validation.

Outputs:

    results/metrics.json
    results/test_predictions.csv
    results/confusion_matrix.csv
    results/validation_trials.csv
    results/model.joblib

Primary Table 5 metric:

    exact Accuracy = number(predicted class == reference class) / N

Additional outputs:
- Macro-F1
- QWK
- classwise precision/recall/F1

------------------------------------------------------------
7. TABLE 5 ROW
------------------------------------------------------------

Run:

    python scripts/06_make_table5_row.py

Expected format:

Published-method reproduction |
VisionMD (2025) |
VisionMD kinematics + LightGBM |
Our HM cohort |
Our held-out HM test set |
HM |
195 clips / X participants |
XX.XX

A more precise evaluation-setting label is:

    Published-tool reproduction

because the original VisionMD publication presented a kinematic-analysis
tool, not a three-class MDS-UPDRS classifier.

------------------------------------------------------------
8. RECOMMENDED MANUSCRIPT DESCRIPTION
------------------------------------------------------------

A concise description is:

"VisionMD was reproduced using its publicly released Hand Movement
kinematic-analysis pipeline. The extracted kinematic features were used
to train a LightGBM classifier on our HM training cohort, with model
selection performed on the validation set and final evaluation conducted
on the same held-out HM test set used for VK-BradyNet."

For strict transparency add:

"Because VisionMD itself does not provide a published three-class severity
classifier, the LightGBM classification stage was introduced only to
enable evaluation using the common exact-accuracy criterion."

------------------------------------------------------------
9. IMPORTANT SHORT-CLIP NOTE
------------------------------------------------------------

VisionMD detects movement cycles and derives cycle-level amplitude, speed,
timing, decay, pause, and hesitation measures. Our study uses 2-s clips.
Some clips may contain too few complete opening-closing cycles to produce
stable decay-related features.

The scripts never loop, duplicate, or extend the video to create artificial
cycles. Feature-extraction failures are logged explicitly.

------------------------------------------------------------
10. REPRODUCIBILITY
------------------------------------------------------------

The setup script stores the exact VisionMD Git commit hash in:

    third_party/VISIONMD_COMMIT.txt

Record this hash in the experiment log before manuscript submission.
