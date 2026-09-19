@echo off
echo FastEval 150-test wrapper
echo.
echo 1. Check dataset:
echo    python scripts\02_check_testset.py
echo.
echo 2. Run official FastEval inference:
echo    python scripts\03_run_official_fasteval.py
echo.
echo 3. Evaluate:
echo    python scripts\04_evaluate_exact_accuracy.py
echo.
echo 4. Generate Table 5 row:
echo    python scripts\05_make_table5_row.py
pause
