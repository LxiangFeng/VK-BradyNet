# Data layout

Put data into:

data/train/0, data/train/1, data/train/2
data/val/0,   data/val/1,   data/val/2
data/test/0,  data/test/1,  data/test/2

or import ZIP archives with scripts/00_import_split.py.

Manifest columns:
filename, relative_path, label, patient_id, side, side_source, side_confidence

Side should be "left" or "right".
