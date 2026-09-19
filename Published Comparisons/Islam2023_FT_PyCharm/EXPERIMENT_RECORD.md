# Experiment record

Paper baseline:
Islam et al. 2023, npj Digital Medicine

Core method retained:
- MediaPipe hand tracking
- thumb/wrist/index angular trajectory
- published kinematic feature extraction
- LightGBM severity regression
- continuous prediction -> nearest severity class

Adaptations for our comparison:
- severity range restricted to 0/1/2
- our fixed patient-level train/val/test split is used instead of paper LOPO CV
- validation set is used for a pre-specified LightGBM hyperparameter grid
- exact Accuracy is calculated on our held-out FT test set
- GUI calls in the official feature extractor are disabled for batch processing
- video duration is read via OpenCV instead of ffprobe (numerical feature definition unchanged)

Record before manuscript submission:
- Official repo commit hash:
- Python version:
- mediapipe version:
- lightgbm version:
- train clips / participants:
- val clips / participants:
- test clips / participants:
- test feature-extraction failures:
- exact Accuracy:
- Macro-F1:
- QWK:
- continuous MAE:
