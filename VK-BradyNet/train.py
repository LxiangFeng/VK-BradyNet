# train.py - Python 版本
import os
from pathlib import Path
import subprocess

# -----------------------------
# 修改为你自己的路径
# -----------------------------
SlowFastDir = Path(r"C:\Users\Administrator\Desktop\SlowFast-main")
TrainDir    = Path(r"C:\Users\Administrator\Desktop\data\train")
ValDir      = Path(r"C:\Users\Administrator\Desktop\data\train")
DataRoot    = Path(r"C:\Users\Administrator\Desktop\data")
YamlFile    = Path(r"C:\Users\Administrator\Desktop\SlowFast-main\pd_mvitv2_s_repro_pack\MVITv2_S_PD_3cls.yaml")
PythonExe   = r"E:\anaconda\envs\YOLO11\python.exe"

# -----------------------------
# 1. 生成 train.csv 和 val.csv
# -----------------------------
#def make_csv(dir_path, csv_file):
    #video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    #with open(csv_file, "w", encoding="utf-8") as f:
#    for cls_dir in dir_path.iterdir():
#           if not cls_dir.is_dir():
#                continue
#           label = int(cls_dir.name)
#            for video in cls_dir.iterdir():
#                if video.suffix.lower() in video_exts:
#                   rel_path = f"{cls_dir.name}/{video.name}"
#                   f.write(f"{rel_path} {label}\n")

#make_csv(TrainDir, DataRoot / "train.csv")
#make_csv(ValDir,   DataRoot / "val.csv")

#print("CSV 文件生成完成！")
#print(f" - {DataRoot / 'train.csv'}")
#print(f" - {DataRoot / 'val.csv'}")

# -----------------------------
# 2. 启动训练
# -----------------------------
cmd = [
    PythonExe,
    str(SlowFastDir / "tools/run_net.py"),
    "--cfg", str(YamlFile),
    "DATA.PATH_TO_DATA_DIR", str(DataRoot),
    "TRAIN.BATCH_SIZE", "4",
    "TRAIN.NUM_GPUS", "1"
]

print("启动训练命令：")
print(" ".join(cmd))
subprocess.run(cmd)