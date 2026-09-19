# train_rgb_keypoint_fusion.py
# Launch RGB + 10D Keypoint feature-level fusion training.

import os
import sys
import subprocess
from pathlib import Path


# 当前文件位置：
# SlowFast-main/pd_mvitv2_s_repro_pack/train_rgb_keypoint_fusion.py
# 所以 parents[1] = SlowFast-main
SlowFastDir = Path(__file__).resolve().parents[1]

ConfigFile = SlowFastDir / "pd_mvitv2_s_repro_pack" / "MVITv2_S_PD_3cls_rgb_keypoint_fusion.yaml"
PythonExe = sys.executable

CheckpointFile = SlowFastDir / "checkpoints" / "MViTv2_S_16x4_k400.pyth"
DataDir = Path(r"C:/Users/Administrator/Desktop/data/data")


def main():
    print("=" * 80)
    print("SlowFast RGB + Keypoint feature-level fusion training (10D)")
    print("=" * 80)
    print(f"Python:     {PythonExe}")
    print(f"Project:    {SlowFastDir}")
    print(f"Config:     {ConfigFile}")
    print(f"Checkpoint: {CheckpointFile}")
    print(f"Data dir:   {DataDir}")
    print("=" * 80)

    if not SlowFastDir.exists():
        raise FileNotFoundError(f"SlowFast project directory does not exist: {SlowFastDir}")

    if not (SlowFastDir / "slowfast").exists():
        raise FileNotFoundError(f"Missing slowfast package directory: {SlowFastDir / 'slowfast'}")

    if not ConfigFile.exists():
        raise FileNotFoundError(f"Config file does not exist: {ConfigFile}")

    if not CheckpointFile.exists():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {CheckpointFile}")

    if not (DataDir / "train.csv").exists():
        raise FileNotFoundError(f"Missing train.csv: {DataDir / 'train.csv'}")

    if not (DataDir / "val.csv").exists():
        raise FileNotFoundError(f"Missing val.csv: {DataDir / 'val.csv'}")

    cmd = [
        PythonExe,
        str(SlowFastDir / "tools" / "run_net.py"),
        "--cfg",
        str(ConfigFile),
    ]

    # 关键修复：给子进程显式传入 PYTHONPATH
    env = os.environ.copy()

    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        env["PYTHONPATH"] = str(SlowFastDir) + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = str(SlowFastDir)

    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)
    print("PYTHONPATH:")
    print(env["PYTHONPATH"])
    print("=" * 80)

    subprocess.run(
        cmd,
        cwd=str(SlowFastDir),
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()