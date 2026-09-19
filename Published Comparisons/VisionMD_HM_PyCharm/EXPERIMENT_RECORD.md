# VisionMD HM reproduction — experiment record

Source method:
Acevedo et al., VisionMD, npj Parkinson's Disease, 2025.

Official public source:
https://github.com/mea-lab/VisionMD-DesktopApp

Reproduction definition:
- HM only (MDS-UPDRS 3.5)
- same patient-level train/val/test split as VK-BradyNet
- VisionMD public HM signal definition
- official public PeakfinderSignalAnalyzer imported directly
- 23/30 scalar kinematic outputs returned by current analyzer (depending on version)
- downstream LightGBM classifier added for common 0/1/2 severity evaluation
- exact Accuracy on the held-out HM test set
- no test-set tuning

Record before manuscript submission:
- VisionMD commit:
- Python:
- MediaPipe:
- LightGBM:
- norm strategy:
- train clips / participants:
- val clips / participants:
- test clips / participants:
- train extraction failures:
- val extraction failures:
- test extraction failures:
- selected LightGBM parameters:
- test Accuracy:
- test Macro-F1:
- test QWK:
- patient-cluster bootstrap 95% CI:
