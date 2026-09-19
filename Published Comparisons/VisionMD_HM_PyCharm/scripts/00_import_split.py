from pathlib import Path
import argparse, zipfile, shutil
from _common import ROOT

VIDEO_EXTS = {".mp4",".mov",".avi",".mkv",".webm"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    zp = Path(args.zip)
    if not zp.exists():
        raise FileNotFoundError(zp)

    dst = ROOT/"data"/args.split
    counts = {0:0,1:0,2:0}
    skipped = []

    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in VIDEO_EXTS:
                continue
            parts = info.filename.strip("/").split("/")
            label = None
            for p in reversed(parts[:-1]):
                if p in {"0","1","2"}:
                    label = int(p)
                    break
            if label is None:
                skipped.append(info.filename)
                continue
            filename = parts[-1]
            out = dst/str(label)/filename
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not args.overwrite:
                continue
            with z.open(info) as src, out.open("wb") as dstf:
                shutil.copyfileobj(src, dstf)
            counts[label] += 1

    print("Imported:", counts)
    if skipped:
        print("Skipped videos with no 0/1/2 parent folder:", len(skipped))

if __name__ == "__main__":
    main()
