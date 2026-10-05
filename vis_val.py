# scripts/vis_from_json.py
import os, cv2, json, argparse
from collections import defaultdict
from pycocotools.coco import COCO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",  default="dataset/image/val")
    ap.add_argument("--ann_file",    default="dataset/annotations/instances_val.json")   # COCO GT json
    ap.add_argument("--det_json",    default="evaluation_results_last.json")   # predictions (xywh in original pixels)
    ap.add_argument("--out_dir",     default="./vis")
    ap.add_argument("--score_thr",   type=float, default=0.05)
    ap.add_argument("--topk",        type=int, default=30)         # <-- NEW
    ap.add_argument("--image_id",    type=int, default=None)       # <-- NEW (optional)
    ap.add_argument("--draw_gt",     action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    coco = COCO(args.ann_file)

    # index predictions by image_id
    with open(args.det_json, "r", encoding="utf-8") as f:
        dets = json.load(f)
    by_img = defaultdict(list)
    for d in dets:
        if d.get("category_id", 1) == 1 and d.get("score", 1.0) >= args.score_thr:
            by_img[int(d["image_id"])].append(d)

    # pick ONE image to draw
    if args.image_id is not None:
        img_ids = [args.image_id]
    else:
        # choose the first image that actually has predictions after thresholding
        img_ids = coco.getImgIds()
        img_ids = [iid for iid in img_ids if len(by_img.get(int(iid), [])) > 0]
        if not img_ids:
            print("No images have predictions above threshold.")
            return

    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(args.images_dir, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read {path}")
            return

        # (optional) draw GT in green
        if args.draw_gt:
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            for a in anns:
                if a.get("iscrowd",0)==1 or a.get("category_id")!=1:
                    continue
                x,y,w,h = a["bbox"]
                x,y,w,h = int(x),int(y),int(w),int(h)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # draw TOP-K predictions (blue), sorted by score desc
        preds = by_img.get(int(img_id), [])
        preds.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        topk = preds[:max(1, args.topk)]

        for rank, d in enumerate(topk, start=1):
            x,y,w,h = d["bbox"]
            x,y,w,h = int(x),int(y),int(w),int(h)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            if "score" in d:
                cv2.putText(img, f"#{rank} {d['score']:.2f}", (x, max(0,y-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                # print(f"#{rank}  score={d['score']:.4f}  bbox={d['bbox']}")

        out_path = os.path.join(args.out_dir, f"{img_id}_top{len(topk)}.jpg")
        cv2.imwrite(out_path, img)
        print(f"wrote {out_path}  (drew top-{len(topk)} / {len(preds)})")

if __name__ == "__main__":
    main()
