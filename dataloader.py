import os
import math
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO

W_FIXED, H_FIXED = 640, 360  # your fixed size

def xywh_to_cxcywh_xy_norm(xywh, W, H):
    x, y, w, h = xywh
    cx = (x + w * 0.5) / W
    cy = (y + h * 0.5) / H
    w  = w / W
    h  = h / H
    return [cx, cy, w, h]

def cxcywh_norm_to_xyxy(cxcywh, W, H):
    cx, cy, w, h = cxcywh
    cx, cy, w, h = cx*W, cy*H, w*W, h*H
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return [x1, y1, x2, y2]

def xyxy_to_cxcywh_norm(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) * 0.5 / W
    cy = (y1 + y2) * 0.5 / H
    w /= W
    h /= H
    return [cx, cy, w, h]

def rotate_points_about_center(points_xy, angle_deg, W, H):
    """points_xy: (N,2) tensor/ndarray in pixel coords"""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    cx, cy = W * 0.5, H * 0.5
    pts = torch.as_tensor(points_xy, dtype=torch.float32)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy
    xr =  c * x - s * y + cx
    yr =  s * x + c * y + cy
    return torch.stack([xr, yr], dim=1)

def rotate_boxes_xyxy(boxes_xyxy, angle_deg, W, H):
    """Rotate 4 corners about (W/2,H/2), then AABB."""
    angle_deg = -angle_deg
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    b = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    p = torch.stack([
        torch.stack([x1, y1], dim=1),
        torch.stack([x2, y1], dim=1),
        torch.stack([x2, y2], dim=1),
        torch.stack([x1, y2], dim=1),
    ], dim=1)  # (N,4,2)
    pr = torch.stack([rotate_points_about_center(p[:,i,:], angle_deg, W, H) for i in range(4)], dim=1)
    x_min, _ = pr[:,:,0].min(dim=1)
    y_min, _ = pr[:,:,1].min(dim=1)
    x_max, _ = pr[:,:,0].max(dim=1)
    y_max, _ = pr[:,:,1].max(dim=1)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

def hflip_boxes_xyxy(boxes_xyxy, W):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    b = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    # flip around vertical axis at W/2: x -> W-1-x
    x1f = W - x2
    x2f = W - x1
    return torch.stack([x1f, y1, x2f, y2], dim=1)

def clip_boxes_xyxy(boxes, W, H):
    if len(boxes) == 0:
        return boxes
    boxes = boxes.clone()
    boxes[:,0] = boxes[:,0].clamp(0, W)
    boxes[:,2] = boxes[:,2].clamp(0, W)
    boxes[:,1] = boxes[:,1].clamp(0, H)
    boxes[:,3] = boxes[:,3].clamp(0, H)
    return boxes

def filter_tiny(boxes_xyxy, min_side=2.0):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, torch.zeros(0, dtype=torch.bool)
    w = (boxes_xyxy[:,2] - boxes_xyxy[:,0]).clamp(min=0)
    h = (boxes_xyxy[:,3] - boxes_xyxy[:,1]).clamp(min=0)
    keep = (w >= min_side) & (h >= min_side)
    return boxes_xyxy[keep], keep

class COCODataset(torch.utils.data.Dataset):
    """
    Minimal single-class COCO dataset loader (category_id=1) with optional aug:
      - rotation (±15°), horizontal flip (p=0.5)
      - fixed output size 640×360 (no resize needed as inputs are already 640×360)
    """
    def __init__(self, images_dir, ann_file, cat_id=1, augment=True, rot_deg=30, hflip_p=0.5):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.getImgIds(catIds=[cat_id]))
        self.cat_id = cat_id
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.augment = augment
        self.rot_deg = rot_deg
        self.hflip_p = hflip_p

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.images_dir, info["file_name"])
        W, H = info["width"], info["height"]  # expected 640×360

        # image
        img = Image.open(path).convert("RGB")

        # annotations → pixel xyxy (easier for geom ops)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.cat_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes_xyxy = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_xyxy.append([x, y, x+w, y+h])
        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.zeros((boxes_xyxy.shape[0],), dtype=torch.long)

        # ---- augment (rotation + hflip), keep size fixed ----
        if self.augment:
            # rotation (expand=False keeps W×H)
            angle = float(torch.empty(1).uniform_(-self.rot_deg, self.rot_deg))


            if abs(angle) > 1e-3:
                # rotate (no fill -> default black outside; we'll crop those regions away)
                img_rot = TF.rotate(
                    img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR,
                    expand=True, center=(W*0.5, H*0.5), fill=(0,0,0)
                )
                newW, newH = img_rot.size  # PIL size: (width, height)

                # compute the center-crop window that brings us back to 640x360
                crop_w, crop_h = W, H
                # top-left of crop in the expanded image (centered)
                left = int(round((newW - crop_w) * 0.5))
                top  = int(round((newH - crop_h) * 0.5))

                # apply same ops to boxes:
                boxes_xyxy = rotate_boxes_xyxy(boxes_xyxy, angle, W, H)

                # finally, crop the image
                img = TF.crop(img_rot, top=top, left=left, height=crop_h, width=crop_w)

            boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, W, H)
            
            if torch.rand(()) < 0.8:  # 80% chance apply random crop
                scale = torch.empty(1).uniform_(0.75, 0.85).item()
                crop_w, crop_h = int(W * scale), int(H * scale)
                x0 = torch.randint(0, W - crop_w + 1, (1,)).item()
                y0 = torch.randint(0, H - crop_h + 1, (1,)).item()

                img = TF.crop(img, top=y0, left=x0, height=crop_h, width=crop_w)
                img = TF.resize(img, (H, W))  # back to 640×360

                if len(boxes_xyxy) > 0:
                    boxes_xyxy[:, [0, 2]] -= x0
                    boxes_xyxy[:, [1, 3]] -= y0
                    boxes_xyxy[:, [0, 2]] *= W / crop_w
                    boxes_xyxy[:, [1, 3]] *= H / crop_h
                boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, W, H)

            # horizontal flip
            if torch.rand(()) < self.hflip_p:
                img = TF.hflip(img)
                if len(boxes_xyxy) > 0:
                    boxes_xyxy = hflip_boxes_xyxy(boxes_xyxy, W)
                
            # clip + filter tiny
            boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, W, H)

            boxes_xyxy, keep = filter_tiny(boxes_xyxy, min_side=2.0)
            labels = labels[keep] if len(keep) == len(labels) else labels[:boxes_xyxy.shape[0]]

        # to tensor + normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)

        # convert boxes back to normalized cx,cy,w,h
        if len(boxes_xyxy) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor([xyxy_to_cxcywh_norm(b.tolist(), W, H) for b in boxes_xyxy], dtype=torch.float32)

        target = {
            "image_id": int(img_id),
            "boxes": boxes,     # normalized cx,cy,w,h
            "labels": labels,   # all zeros for single class
        }
        return img, target

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)
