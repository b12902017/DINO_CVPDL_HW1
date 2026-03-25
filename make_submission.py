import json
import csv
from pathlib import Path

# paths
RAW_JSON = "vis_test/predictions_raw.json"   # produced by infer_draw.py
OUT_CSV  = "submission.csv"

# if your filenames are like '1.jpg', we extract the numeric part
def get_image_id(filename: str) -> int:
    stem = Path(filename).stem
    try:
        return int(stem)
    except:
        # fallback: assign incremental id later
        return None

# load detections
with open(RAW_JSON, "r", encoding="utf-8") as f:
    dets = json.load(f)

# group by file_name
from collections import defaultdict
grouped = defaultdict(list)
for d in dets:
    fname = d["file_name"]
    x1, y1, x2, y2 = d["bbox_xyxy"]
    conf = d["score"]
    cls  = 0  # pig
    w = x2 - x1
    h = y2 - y1
    grouped[fname].append((conf, x1, y1, w, h, cls))

# make rows
rows = []
for i, (fname, preds) in enumerate(sorted(grouped.items()), start=1):
    preds.sort(key=lambda x: -x[0])  # sort by confidence (optional)
    pred_str = " ".join(
        f"{conf:.4f} {x1} {y1} {w} {h} {cls}" for conf,x1,y1,w,h,cls in preds
    )
    image_id = get_image_id(fname) or i
    rows.append((image_id, pred_str))

# write CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_ID", "PredictionString"])
    writer.writerows(rows)

print(f"✅ Submission saved to: {OUT_CSV}")
