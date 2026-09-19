@echo off
echo Islam et al. 2023 FT reproduction
echo.
echo Recommended order:
echo   python scripts\01_setup_official_repo.py
echo   python scripts\02_build_manifest.py --split train
echo   python scripts\02_build_manifest.py --split val
echo   python scripts\02_build_manifest.py --split test
echo   python scripts\03_autodetect_side.py --split train
echo   python scripts\03_autodetect_side.py --split val
echo   python scripts\03_autodetect_side.py --split test
echo   python scripts\04_extract_official_features.py --split train
echo   python scripts\04_extract_official_features.py --split val
echo   python scripts\04_extract_official_features.py --split test
echo   python scripts\05_train_evaluate_lightgbm.py --mode val_tune
echo   python scripts\06_make_table5_row.py
pause
