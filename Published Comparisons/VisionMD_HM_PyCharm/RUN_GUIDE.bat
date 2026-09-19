@echo off
echo VisionMD HM reproduction
echo.
echo 1) pip install -r requirements.txt
echo 2) python scripts\01_setup_official_visionmd.py
echo 3) python scripts\02_download_hand_model.py
echo 4) python scripts\03_build_manifest.py --split train
echo 5) python scripts\03_build_manifest.py --split val
echo 6) python scripts\03_build_manifest.py --split test
echo 7) Edit side/ROI in data\manifests if real metadata are available
echo 8) python scripts\04_extract_visionmd_hm_features.py --split train
echo 9) python scripts\04_extract_visionmd_hm_features.py --split val
echo 10) python scripts\04_extract_visionmd_hm_features.py --split test
echo 11) python scripts\05_train_evaluate.py --mode val_tune
echo 12) python scripts\06_make_table5_row.py
echo 13) python scripts\07_patient_cluster_bootstrap.py
pause
