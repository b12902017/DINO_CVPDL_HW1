import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.utils import draw_bounding_boxes
from PIL import Image

from dataloader import COCODataset, collate_fn

# --- import your dataset code here ---
# from your_file import COCODataset, collate_fn   # <- adjust the import

W, H = 640, 360  # your fixed size

def denorm_to_uint8(img, mean, std):
    """img: FloatTensor[3,H,W], normalized. Return: uint8 tensor [3,H,W] in 0..255."""
    x = img.clone()
    for c in range(3):
        x[c] = x[c] * std[c] + mean[c]
    x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
    return x

def cxcywh_norm_to_xyxy_tensor(boxes_norm, W, H):
    """boxes_norm: Tensor[N,4] in [cx,cy,w,h] normalized. Return: Tensor[N,4] xyxy in pixels."""
    if boxes_norm.numel() == 0:
        return boxes_norm.reshape(0,4)
    cxcywh = boxes_norm.clone()
    cxcywh[:, 0] *= W
    cxcywh[:, 1] *= H
    cxcywh[:, 2] *= W
    cxcywh[:, 3] *= H
    x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2.0
    y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2.0
    x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2.0
    y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2.0
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    # clip just in case
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, W)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, H)
    return boxes_xyxy

def visualize_loader(dataset, out_dir="debug_vis", max_images=16):
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    mean, std = dataset.mean, dataset.std

    saved = 0
    for imgs, targets in loader:
        # batch_size = 1 by design
        img = imgs[0]                               # FloatTensor[3,H,W], normalized
        target = targets[0]
        boxes_norm = target["boxes"]                # [N,4] normalized cx,cy,w,h

        # denorm to uint8 for drawing
        img_u8 = denorm_to_uint8(img, mean, std)    # [3,H,W], uint8

        # convert boxes to xyxy pixels for drawing
        boxes_xyxy = cxcywh_norm_to_xyxy_tensor(boxes_norm, W, H)

        # draw (if no boxes, it'll just return the image)
        img_drawn = draw_bounding_boxes(
            image=img_u8,
            boxes=boxes_xyxy,
            labels=[str(i) for i in range(boxes_xyxy.shape[0])],  # optional
            width=2
        )

        # save
        pil_img = TF.to_pil_image(img_drawn)
        pil_img.save(os.path.join(out_dir, f"sample_{saved:03d}.jpg"))

        saved += 1
        if saved >= max_images:
            break

    print(f"Saved {saved} debug images to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    # toggle augment=True to visualize your rotation/flip path
    ds = COCODataset(
        images_dir="dataset/image/train/",
        ann_file="dataset/annotations/instances_train.json",
        cat_id=1,
        augment=True,      # <- make sure your dataset supports this flag from our earlier patch
        rot_deg=15,
        hflip_p=0.5
    )
    visualize_loader(ds, out_dir="debug_vis", max_images=12)
