from pathlib import Path
import argparse
import zipfile
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="ZIP containing class folders 0/1/2 anywhere in the path")
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    zp = Path(args.zip)
    if not zp.exists():
        raise FileNotFoundError(zp)

    dst_root = ROOT / "data" / args.split
    counts = {0:0,1:0,2:0}
    skipped = []

    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".mp4",".mov",".avi",".mkv",".webm")):
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
            out = dst_root / str(label) / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not args.overwrite:
                continue
            with z.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[label] += 1

    print("Imported:", counts)
    if skipped:
        print(f"Skipped {len(skipped)} video files because no 0/1/2 class folder was found.")
    print("Next: python scripts/02_build_manifest.py --split", args.split)

if __name__ == "__main__":
    main()
