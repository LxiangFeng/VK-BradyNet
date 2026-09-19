# train_keypoint_only.py
# Launch Keypoint-only baseline training.
# Ablation experiment: Keypoint branch only.

import os
import sys
import subprocess
from pathlib import Path


# SlowFast-main directory
SlowFastDir = Path(__file__).resolve().parents[1]

# Keypoint-only YAML configuration
ConfigFile = (
    SlowFastDir
    / "pd_mvitv2_s_repro_pack"
    / "MVITv2_S_PD_3cls_keypoint_only.yaml"
)

PythonExe = sys.executable


def main():

    print("=" * 80)
    print("SlowFast Keypoint-only baseline training")
    print("=" * 80)

    print(f"Python:     {PythonExe}")
    print(f"Project:    {SlowFastDir}")
    print(f"Config:     {ConfigFile}")
    print("=" * 80)

    if not SlowFastDir.exists():
        raise FileNotFoundError(
            f"SlowFast project directory does not exist: {SlowFastDir}"
        )

    if not (SlowFastDir / "slowfast").exists():
        raise FileNotFoundError(
            f"Missing slowfast package directory: {SlowFastDir / 'slowfast'}"
        )

    if not ConfigFile.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {ConfigFile}"
        )

    cmd = [
        PythonExe,
        str(SlowFastDir / "tools" / "run_net.py"),
        "--cfg",
        str(ConfigFile),
    ]

    env = os.environ.copy()

    old_pythonpath = env.get("PYTHONPATH", "")

    if old_pythonpath:
        env["PYTHONPATH"] = (
            str(SlowFastDir)
            + os.pathsep
            + old_pythonpath
        )
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
