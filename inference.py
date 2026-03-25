import os, json
from pathlib import Path
from typing import List, Tuple
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

# ---- import your model / utility ----
from main import SimpleDETR, cxcywh_to_xyxy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

@torch.no_grad()
def preprocess(img_pil: Image.Image) -> Tuple[torch.Tensor, Tuple[int,int]]:
    W, H = img_pil.size  # expect 640x360
    x = TF.to_tensor(img_pil)
    x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return x.unsqueeze(0), (W, H)

@torch.no_grad()
def postprocess_topk(
    boxes_cxcywh: torch.Tensor,  # (Q,4) normalized
    scores: torch.Tensor,        # (Q,)
    W: int, H: int,
    topk: int = 100
) -> List[Tuple[List[int], float]]:
    K = min(topk, scores.numel())
    vals, idxs = scores.topk(K)
    boxes = boxes_cxcywh[idxs]

    # normalized → pixel xyxy using actual W,H
    xyxy = cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], device=boxes.device)
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)

    out = []
    for b, s in zip(xyxy, vals):
        x1, y1, x2, y2 = b.round().to(torch.int32).tolist()
        if x2 > x1 and y2 > y1:  # keep only valid
            out.append(([x1, y1, x2, y2], float(s)))
    return out

def map_boxes_back(pred_boxes, crop_x0, crop_y0, crop_w, crop_h, orig_w, orig_h):
    # pred_boxes: Tensor[N,4] normalized cx,cy,w,h (in cropped+resized coords)
    boxes_xyxy = cxcywh_to_xyxy(pred_boxes)  # still normalized (0–1)
    boxes_xyxy[:, [0, 2]] *= crop_w  # scale to cropped pixel coords
    boxes_xyxy[:, [1, 3]] *= crop_h
    boxes_xyxy[:, [0, 2]] += crop_x0
    boxes_xyxy[:, [1, 3]] += crop_y0
    return boxes_xyxy

@torch.no_grad()
def run_folder(
    model,
    images_dir: str,
    out_dir: str,
    device: str = "cuda",
    topk: int = 100,
    save_json: bool = True
):
    model.eval().to(device)
    os.makedirs(out_dir, exist_ok=True)

    det_dump = []
    img_paths = []
    img_paths.extend(Path(images_dir).rglob("*.jpg"))
    img_paths = sorted(img_paths)

    for p in tqdm(img_paths, desc="Infer", ncols=100):

        img = Image.open(p).convert("RGB")
        W, H = img.size  # (640, 360)
        '''
        # 2. Define crop region (bottom-left 88%)
        crop_x0, crop_y0 = 0, int(H * 0.1)
        crop_w, crop_h = int(W * 0.85), int(H * 0.9)

        # 3. Crop & resize back to original size
        cropped = img.crop((crop_x0, crop_y0, crop_x0 + crop_w, crop_y0 + crop_h))
        img = cropped.resize((W, H), Image.BILINEAR)
        '''
        # 4. Preprocess for model (tensor in [0,1])
        x, _ = preprocess(img)
        x = x.to(device)

        # 5. Forward pass
        outputs = model(x)
        pred_boxes = outputs['pred_boxes'][0].cpu()  # normalized (cx,cy,w,h)
        logits = outputs['pred_logits'][0].softmax(-1)
        scores = logits[:, :-1].max(-1)[0]

        # 6. Map boxes back to original coordinates
        # pred_boxes = map_boxes_back(pred_boxes, crop_x0, crop_y0, crop_w, crop_h, W, H)

        # 7. Postprocess (select top-K by score)

        boxes_xyxy = cxcywh_to_xyxy(pred_boxes)  # still normalized (0–1)
        boxes_xyxy = boxes_xyxy * torch.tensor([W, H, W, H], device=boxes_xyxy.device)

        scores = scores.cpu()
        boxes_xyxy = boxes_xyxy.cpu()

        sorted_idx = torch.argsort(scores, descending=True)
        boxes_xyxy = boxes_xyxy[sorted_idx]
        scores = scores[sorted_idx]

        # Optionally truncate to top-K
        if topk is not None and topk < len(scores):
            boxes_xyxy = boxes_xyxy[:topk]
            scores = scores[:topk]

        dets = torch.cat([boxes_xyxy, scores[:, None]], dim=1)  # (K,5)
        # draw
        vis = img.copy()
        drw = ImageDraw.Draw(vis)
        rank = 1
        for x1,y1,x2,y2,sc in dets:
            # color varies with score for readability
            c = int(255 * max(0.0, min(1.0, sc)))
            drw.rectangle([x1, y1, x2, y2], outline=(255-c, c, 0), width=2)
            drw.text((x1+2, y1+2), f"{rank}", fill=(255,255,255))
            rank += 1
        vis.save(Path(out_dir)/p.name)

        # raw dump (we’ll convert to your final format next)
        for x1,y1,x2,y2,sc in dets:
            x1, y1, x2, y2, sc = x1.item(), y1.item(), x2.item(), y2.item(), sc.item()
            det_dump.append({
                "file_name": p.name,
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": sc,
                "category_id": 1
            })

    if save_json:
        with open(Path(out_dir)/"predictions_raw.json", "w", encoding="utf-8") as f:
            json.dump(det_dump, f, ensure_ascii=False, indent=2)

    print(f"saved visuals -> {out_dir}")
    if save_json:
        print(f"raw detections -> {Path(out_dir)/'predictions_raw.json'}")

# ------------- CLI -------------
if __name__ == "__main__":
    import argparse
    from main import SimpleDETR  # uses your class as-is

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="testdata/img")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth")
    parser.add_argument("--out", type=str, default="vis_test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--pretrained-backbone", action="store_true")
    args = parser.parse_args()

    model = SimpleDETR(
        num_classes=args.num_classes,
        num_queries=args.queries,
        pretrained_backbone=args.pretrained_backbone
    )
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=True)

    run_folder(model, args.images, args.out, device=args.device, topk=args.topk, save_json=True)
