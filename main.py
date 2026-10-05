"""
A minimal, readable DETR/DINO-style implementation in one file.
- Pure PyTorch + torchvision only (for backbone and box ops)
- Backbone can be pretrained (ResNet-50 default)
- Hungarian matching, SetCriterion losses (cls CE + L1 + GIoU)
- Optional DINO-style denoising (very small, single-group variant)

This is designed as a teaching/reference script: concise over feature-complete.
It should run on a small toy dataset; for COCO you must write a Dataset that
returns targets in the expected format (see TinyToyDataset for example).

Usage (toy):
  python simple_detr_dino.py --epochs 2 --toy

Usage (custom dataset):
  Implement your torch.utils.data.Dataset to return (image, target) pairs:
    image: Tensor[3,H,W] in [0,1]
    target: {
      'boxes': Tensor[N,4] in [0,1] (cx,cy,w,h normalized),
      'labels': Tensor[N] in [0,num_classes-1]
    }
  Then plug into the DataLoader below.
"""
import math
import random
import argparse
import os, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.ops import box_iou, generalized_box_iou
import scipy.optimize
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataloader import COCODataset  # or your custom Dataset

# ----------------------------
# Positional Encoding (sin-cos 2D)
# ----------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int = 256, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: (B, H, W) with True for padding, False for valid pixels
        returns: pos (B, d_model, H, W)
        """
        not_mask = (~mask).float()  # 1 for valid
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps)
        x_embed = x_embed / (x_embed[:, :, -1:] + eps)

        dim_t = self.temperature ** (2 * torch.div(torch.arange(self.d_model // 2, device=mask.device), 2, rounding_mode='floor') / (self.d_model))
        pos_x = x_embed[:, :, :, None] / dim_t  # (B,H,W,C)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# ----------------------------
# Small MLP head for box regression
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Minimal DETR
# ----------------------------
class SimpleDETR(nn.Module):
    def __init__(self, num_classes: int, num_queries: int = 100, hidden_dim: int = 256,
                 nheads: int = 8, enc_layers: int = 3, dec_layers: int = 3,
                 pretrained_backbone: bool = True):
        super().__init__()
        # Backbone (ResNet-50) → 2048 channels at stride 32
        print("Loading ResNet-50 backbone, pretrained =", pretrained_backbone)
        backbone = torchvision.models.resnet50(weights=(
            torchvision.models.ResNet50_Weights.DEFAULT if pretrained_backbone else None))
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        self.pos_enc = PositionalEncoding2D(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Identity()  # conv already to hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4)

        self.label_embed = nn.Embedding(num_classes, hidden_dim)
        self.box_embed = MLP(4, hidden_dim, hidden_dim)

    def _get_src_and_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        # x: (B,3,H,W) in [0,1]
        feats = self.backbone(x)  # (B,2048,h,w)
        h, w = feats.shape[-2:]
        src = self.conv(feats)  # (B,C,h,w)
        mask = torch.zeros((x.size(0), h, w), dtype=torch.bool, device=x.device)  # no padding for 
        pos = self.pos_enc(mask)  # (B,C,h,w)
        return src, pos, mask, h, w

    def forward(self, x: torch.Tensor, dn_targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        src, pos, mask, h, w = self._get_src_and_mask(x)
        # Flatten HW → S
        src_flat = src.flatten(2).permute(2, 0, 1)      # (S,B,C)
        pos_flat = pos.flatten(2).permute(2, 0, 1)      # (S,B,C)
        mask_flat = mask.flatten(1)                     # (B,S)

        memory = self.encoder(src_flat + pos_flat, src_key_padding_mask=mask_flat)

        B = x.size(0)
        query_emb = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries,B,C)
        tgt = torch.zeros_like(query_emb) 

        num_dn = 0
        if dn_targets is not None:
            num_dn = max(len(t['labels']) for t in dn_targets) if dn_targets else 0 

        if num_dn > 0:
            dn_embs = []
            for t in dn_targets:
                n = len(t['labels'])
                if n == 0:
                    dn_embs.append(torch.zeros((num_dn, query_emb.size(-1)), device=x.device))
                    continue
                labels = t['labels']
                boxes = t['boxes']
                noise = torch.randn_like(boxes) * 0.2
                noisy_boxes = (boxes + noise).clamp(0, 1)
                emb = self.label_embed(labels) + self.box_embed(noisy_boxes)
                if n < num_dn:
                    pad = torch.zeros((num_dn - n, query_emb.size(-1)), device=x.device)
                    emb = torch.cat([emb, pad], 0)
                dn_embs.append(emb)
            dn_tgt = torch.stack(dn_embs, dim=1)  # max_gt, B, C
            dn_query_emb = torch.zeros_like(dn_tgt)

            tgt_all = torch.cat([dn_tgt, tgt], dim=0)                   # (num_dn+Q,B,C)
            query_all = torch.cat([dn_query_emb, query_emb], dim=0)     # (num_dn+Q,B,C)
        else:
            tgt_all = tgt
            query_all = query_emb

        hs = self.decoder(tgt_all + query_all, memory + pos_flat, memory_key_padding_mask=mask_flat)
        # hs: (num_queries,B,C)
        hs = hs.transpose(0, 1)  # (B, num_dn+Q, C)

        if num_dn > 0:
            dn_hs = hs[:, :num_dn, :]
            main_hs = hs[:, num_dn:, :]
            dn_logits = self.class_embed(dn_hs)
            dn_boxes = self.bbox_embed(dn_hs).sigmoid()
            dn = {"dn_pred_logits": dn_logits, "dn_pred_boxes": dn_boxes, "dn_targets": dn_targets}
        else:
            main_hs = hs
            dn = None

        outputs_class = self.class_embed(main_hs)  # (B,num_queries,num_classes+1)
        outputs_bbox = self.bbox_embed(main_hs).sigmoid().clamp(0,1)  # normalized (cx,cy,w,h)
        return {"pred_logits": outputs_class, "pred_boxes": outputs_bbox, "dn": dn}

# ----------------------------
# Utility: box conversion once
# ----------------------------

def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

# ----------------------------
# Hungarian matcher
# ----------------------------
@dataclass
class MatcherConfig:
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0

class HungarianMatcher(nn.Module):
    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # (B,Q,C+1)
        out_bbox = outputs["pred_boxes"]               # (B,Q,4)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]['labels']  # (N,)
            tgt_bbox = targets[b]['boxes']  # (N,4)
            device = out_bbox.device
            if tgt_bbox.numel() == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue

            # Classification cost: -P(class)
            cost_class = -out_prob[b][:, tgt_ids]

            # L1 bbox cost
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

            # GIoU cost (need (x1y1x2y2))
            giou = generalized_box_iou(cxcywh_to_xyxy(out_bbox[b]), cxcywh_to_xyxy(tgt_bbox))
            cost_giou = -giou

            C = self.cfg.cost_class * cost_class + self.cfg.cost_bbox * cost_bbox + self.cfg.cost_giou * cost_giou
            if not torch.isfinite(C).all():
                print(f"[Warning] Invalid cost entries (NaN/Inf) detected. Replacing with large values.")
                C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
            C = C.detach().cpu()
            idx_q, idx_t = scipy.optimize.linear_sum_assignment(C)
            indices.append((torch.as_tensor(idx_q, dtype=torch.int64, device=device), torch.as_tensor(idx_t, dtype=torch.int64, device=device)))
        return indices

# ----------------------------
# SetCriterion loss
# ----------------------------
class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher,
                 weight_dict: Dict[str, float], eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']  # (B,Q,C+1)
        bs, num_queries = src_logits.shape[:2]
        device = self.empty_weight.device
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J].to(device) for t, (_, J) in zip(targets, indices)], dim=0)
        target_classes = torch.full((bs, num_queries), self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        device = self.empty_weight.device
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][J].to(device) for t, (_, J) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / max(src_boxes.size(0), 1)
        giou = generalized_box_iou(cxcywh_to_xyxy(src_boxes), cxcywh_to_xyxy(target_boxes))
        loss_giou = (1 - giou.diag()).sum() / max(src_boxes.size(0), 1)
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        weighted = sum(self.weight_dict[k] * v for k, v in losses.items())
        return weighted, losses

# ----------------------------
# Training utilities (tiny optimizations)
# ----------------------------

def train_one_epoch(model, criterion, optimizer, loader, device, dn_wrapper=None, dn_weight=1.0, scaler=None, clip_norm: float = 0.1, ema=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for images, targets in tqdm(loader, desc="Training", ncols=100):
        images = images.to(device)
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            if dn_wrapper is None:
                outputs = model(images)
                total, losses = criterion(outputs, targets)
            else:
                outputs, dn = dn_wrapper(images, targets)
                total, losses = criterion(outputs, targets)
                if dn is not None:
                    dn_logits, dn_boxes, dn_targets = dn['dn_pred_logits'], dn['dn_pred_boxes'], dn['dn_targets']
                    dn_t_labels = torch.cat([t['labels'] for t in dn_targets]) if dn_targets else torch.empty(0, dtype=torch.long, device=device)
                    dn_t_boxes = torch.cat([t['boxes'] for t in dn_targets]) if dn_targets else torch.empty(0, 4, device=device)
                    dn_p_logits = torch.cat([dn_logits[i, :t['labels'].numel()] for i, t in enumerate(dn_targets)]) if dn_targets else torch.empty(0, dn_logits.size(-1), device=device)
                    dn_p_boxes = torch.cat([dn_boxes[i, :t['labels'].numel()] for i, t in enumerate(dn_targets)]) if dn_targets else torch.empty(0, 4, device=device)
                    if dn_t_labels.numel() > 0:
                        dn_ce = F.cross_entropy(dn_p_logits, dn_t_labels)
                        dn_l1 = F.l1_loss(dn_p_boxes, dn_t_boxes)
                        giou = generalized_box_iou(cxcywh_to_xyxy(dn_p_boxes), cxcywh_to_xyxy(dn_t_boxes))
                        dn_giou = (1 - giou.diag()).mean()
                        total = total + dn_weight * (dn_ce + dn_l1 + dn_giou)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
                if dn_wrapper is None:
                    outputs = model(images)
                    total, losses = criterion(outputs, targets)
                else:
                    outputs, dn = dn_wrapper(images, targets)
                    total, losses = criterion(outputs, targets)
                    if dn is not None:
                        dn_logits, dn_boxes, dn_targets = dn['dn_pred_logits'], dn['dn_pred_boxes'], dn['dn_targets']
                        dn_t_labels = torch.cat([t['labels'] for t in dn_targets]) if dn_targets else torch.empty(0, dtype=torch.long, device=device)
                        dn_t_boxes = torch.cat([t['boxes'] for t in dn_targets]) if dn_targets else torch.empty(0, 4, device=device)
                        dn_p_logits = torch.cat([dn_logits[i, :t['labels'].numel()] for i, t in enumerate(dn_targets)]) if dn_targets else torch.empty(0, dn_logits.size(-1), device=device)
                        dn_p_boxes = torch.cat([dn_boxes[i, :t['labels'].numel()] for i, t in enumerate(dn_targets)]) if dn_targets else torch.empty(0, 4, device=device)
                        if dn_t_labels.numel() > 0:
                            dn_ce = F.cross_entropy(dn_p_logits, dn_t_labels)
                            dn_l1 = F.l1_loss(dn_p_boxes, dn_t_boxes)
                            giou = generalized_box_iou(cxcywh_to_xyxy(dn_p_boxes), cxcywh_to_xyxy(dn_t_boxes))
                            dn_giou = (1 - giou.diag()).mean()
                            total = total + dn_weight * (dn_ce + dn_l1 + dn_giou)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
        total_loss += total.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0

# ----------------------------
# Evaluation utilities
# ----------------------------
def evaluate(model, criterion, loader, device, coco, cat_id=1):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    coco_dt = []
    coco.dataset.setdefault("info", {})
    coco.dataset.setdefault("licenses", [])
    if "categories" not in coco.dataset or len(coco.dataset["categories"]) == 0:
        coco.dataset["categories"] = [{"id": 1, "name": "object"}]
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            outputs = model(images)
            total, losses = criterion(outputs, targets)
            total_loss += total.item()
            num_batches += 1
            
            # Collect predictions for mAP
            for i in range(images.size(0)):
                logits = outputs['pred_logits'][i].softmax(-1)  # (Q, C+1)
                boxes = outputs['pred_boxes'][i]  # (Q, 4)
                scores = logits[:, :-1].max(-1)[0]  # (Q,)
                labels = logits[:, :-1].max(-1)[1]  # (Q,)
                
                # Filter predictions (score > 0.5)
                '''
                mask = scores > 0.5
                if mask.sum() == 0:
                    continue
                '''

                pred_boxes = boxes  # (N', 4) in cxcywh
                pred_scores = scores  # (N')
                pred_labels = labels  # (N')
                
                # Convert boxes to xyxy in pixel coordinates
                img_id = targets[i]['image_id']
                img_info = coco.loadImgs(img_id)[0]
                W, H = img_info['width'], img_info['height']
                pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes) * torch.tensor([W, H, W, H], device=device)

                def xyxy_to_xywh(xyxy):
                    # xyxy: Tensor[N,4] on device
                    x1, y1, x2, y2 = xyxy.unbind(-1)
                    w = x2 - x1
                    h = y2 - y1
                    return torch.stack([x1, y1, w, h], dim=-1)
                
                for box, score, label in zip(xyxy_to_xywh(pred_boxes_xyxy), pred_scores, pred_labels):
                    coco_dt.append({
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': box.cpu().numpy().tolist(),
                        'score': score.cpu().item()
                    })

    dt_file = "evaluation_results_last.json"
    with open(dt_file, "w", encoding="utf-8") as f:
        json.dump(coco_dt, f)
    
    # Compute mAP
    coco_dt = coco.loadRes(coco_dt)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.catIds = [cat_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]  # AP@[0.5:0.95]
    
    return total_loss / num_batches, mAP


class DenoisingWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, images, targets):
        outputs = self.model(images, dn_targets=targets)
        dn = outputs.pop("dn", None)
        return outputs, dn
    
class ModelEMA:
    def __init__(self, model, decay=0.999):
        # Make a copy of the model for accumulating moving averages
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            v = esd[k]
            if torch.is_floating_point(v):
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                esd[k] = msd[k]  # buffers (e.g., running stats)

    def to(self, device):
        self.ema.to(device)
        return self

def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=1e-5, help='smaller LR for backbone (classic DETR practice)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--pretrained_backbone', default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dino', action='store_true', help='Enable simple DINO-style denoising training')
    parser.add_argument('--amp', action='store_true', help='mixed precision training')
    parser.add_argument('--clip-norm', type=float, default=0.1)
    parser.add_argument('--images-dir', type=str, default='dataset/image/train', help='Path to COCO images directory')
    parser.add_argument('--ann-file', type=str, default='dataset/annotations/instances_train.json', help='Path to COCO annotations JSON')
    parser.add_argument('--val-images-dir', type=str, default='dataset/image/val')
    parser.add_argument('--val-ann-file', type=str, default='dataset/annotations/instances_val.json')
    parser.add_argument('--checkpoint-dir', type=str, default='./weights', help='Directory to save model weights')
    parser.add_argument('--log-file', type=str, default='training_log.txt', help='File to log training metrics')
    parser.add_argument('--checkpoint', action='store_true', help='Load from checkpoint if available')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model = SimpleDETR(num_classes=args.num_classes, num_queries=args.queries,
                       pretrained_backbone=args.pretrained_backbone).to(device)
    if args.checkpoint:
        print("Loading from checkpoint...")
        state_dict = torch.load('weights/hist/aug_100ep.pth', map_location=device)
        model.load_state_dict(state_dict)
    
    ema = ModelEMA(model, decay=0.999).to(device)

    dn_wrapper = DenoisingWrapper(model) if args.dino else None

    matcher = HungarianMatcher(MatcherConfig())
    criterion = SetCriterion(num_classes=args.num_classes, matcher=matcher,
                             weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}).to(device)

    dataset = COCODataset(images_dir=args.images_dir, ann_file=args.ann_file, cat_id=1)
    val_dataset = COCODataset(images_dir=args.val_images_dir, ann_file=args.val_ann_file, cat_id=1, augment=False)
    coco_gt = COCO(args.val_ann_file)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Optimizer with two LR groups (backbone smaller)
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': other_params, 'lr': args.lr}
    ], lr=args.lr, weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type=='cuda')

    best_mAP = 0.0
    with open(args.log_file, 'w') as f:
        f.write("Epoch,Train Loss,Val Loss,mAP\n")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, dn_wrapper=dn_wrapper, scaler=scaler, clip_norm=args.clip_norm, ema=ema)
        
        #Evaluate
        val_loss, mAP = evaluate(ema.ema, criterion, val_loader, device, coco_gt, cat_id=1)

        with open(args.log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{mAP:.4f}\n")

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(ema.ema.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
        
        torch.save(ema.ema.state_dict(), os.path.join(args.checkpoint_dir, 'last_model.pth'))

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, mAP={mAP:.4f}")

if __name__ == '__main__':
    main()