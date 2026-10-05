# Pig detection with a DETR / DINO-style model

This repository is a single-class object detection experiment for locating pigs in images. It implements a small DETR detector in PyTorch, with optional DINO-style denoising during training. The workflow covers COCO-format data loading, augmentation, training, validation, drawing detections, and exporting a submission CSV.

The model is a custom, simplified implementation rather than the full official DINO architecture. This README explains the existing code and how to run a similar experiment with your own data; it is not an exact reproduction recipe for the original run.

## What was implemented

- A ResNet-50 backbone, optionally initialized with ImageNet weights, followed by a 256-dimensional feature projection and 2D positional encoding.
- A Transformer with three encoder layers, three decoder layers, eight attention heads, and 100 learned object queries by default.
- A pig/background classification head and a bounding-box regression head.
- Hungarian matching and classification, L1 box, and generalized IoU losses, weighted `1 : 5 : 2`.
- Optional denoising queries built from ground-truth labels and noisy boxes (`--dino`). This is a small single-group variant, without the full DINO training machinery.
- AdamW with separate backbone and detector learning rates, gradient clipping, optional mixed precision, and an exponential moving average (EMA) model for validation and saved weights.
- COCO bounding-box AP evaluation after each epoch.

The local annotations inspected for this project describe 1,076 training images with 32,840 pig boxes and 190 validation images with 5,779 boxes. All are 640 × 360 pixels. The existing 200-epoch `training_log.txt` records a highest validation AP@[0.50:0.95] of **0.4761** at epoch 189. The log does not record the command, dependency versions, or whether denoising was enabled, so it cannot establish the exact configuration behind that result.

## Repository files

| File | Purpose |
| --- | --- |
| `main.py` | Model, matching, losses, training loop, EMA, and COCO evaluation. |
| `dataloader.py` | Single-class COCO loader, box conversions, and training augmentations. |
| `data_aug_checker.py` | Draws augmented training samples and their boxes for inspection. |
| `vis_val.py` | Draws validation predictions, optionally alongside ground truth. |
| `inference.py` | Loads a checkpoint, predicts on a folder of JPG images, and saves images and raw JSON. |
| `make_submission.py` | Converts the inference JSON into `submission.csv`. |
| `training_log.txt` | Historical epoch-level training loss, validation loss, and AP. |

Datasets, weights, and generated outputs are ignored by Git. They may exist in a local working copy but are not required to be included in a clone. There is no dataset download, annotation conversion, or train/validation split script; supply these yourself. The original dataset source is not documented in the code.

## Environment

Use a Python environment with compatible PyTorch and torchvision installations. A CUDA GPU is recommended for training; the code also accepts `--device cpu`. Exact versions from the original experiment are not recorded, and dependencies are not pinned.

For example, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision scipy pycocotools pillow tqdm opencv-python
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Choose PyTorch packages appropriate to your CUDA setup if using a GPU. The code uses APIs such as `torch.amp.GradScaler` and torchvision's weights enums, so older installations may need updating. OpenCV is imported by inference and used by the validation visualization script.

Training initializes ResNet-50 with pretrained weights by default and may download them on first use. No full detector checkpoint is downloaded automatically.

## 1. Prepare data

The default layout is:

```text
dataset/
  image/
    train/                         # training images
    val/                           # validation images
  annotations/
    instances_train.json           # COCO ground truth
    instances_val.json             # COCO ground truth
testdata/
  img/                             # images for folder inference
weights/                           # created during training
```

Each COCO annotation file needs `images`, `annotations`, and `categories`. Image entries specify `id`, `file_name`, `width`, and `height`; annotation entries specify `id`, `image_id`, `category_id`, `bbox`, `area`, and `iscrowd`. Boxes use COCO pixel coordinates `[x, y, width, height]`. For example:

```json
{
  "images": [{"id": 1, "file_name": "00000001.jpg", "width": 640, "height": 360}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
    "bbox": [100, 80, 120, 60], "area": 7200, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "pig"}]
}
```

`file_name` is resolved relative to the corresponding image directory. The loader selects COCO category **1** and maps it to model label **0**. It selects image IDs associated with that category, so images without annotations for category 1 are excluded. Use nonempty training and validation splits.

For a similar single-class experiment, annotate your target object as category 1. Changing `--num-classes` alone does not make the current data loader and evaluation pipeline support multiple classes.

Prepare consistently sized images, ideally the original 640 × 360 size, and ensure the annotation dimensions match the actual files. Training stacks images directly into batches and does not generally resize or pad them. If you resize your own images, scale the annotation boxes too.

Training augmentation uses random rotation up to ±30°, a random crop with 80% probability (75–85% of the original dimensions, resized back), and horizontal flipping with probability 0.5. Boxes are transformed, clipped, and filtered if either side is under two pixels. Both training and inference use ImageNet mean/std normalization.

Optionally inspect augmented samples:

```bash
python data_aug_checker.py
```

This writes 12 samples to `debug_vis/`. The checker uses hardcoded default data paths, 640 × 360 box drawing dimensions, and ±15° rotation, so its rotation setting differs from training.

## 2. Train and evaluate

Run from the repository root:

```bash
python main.py --device cuda --epochs 200 --batch-size 4 --dino --amp
```

Omit `--dino` for the base DETR training path. Omit `--amp` to disable mixed precision. For a small CPU trial, use `--device cpu --epochs 2 --batch-size 1`; this still requires real training and validation data.

To use other data locations:

```bash
python main.py \
  --images-dir /path/to/train/images \
  --ann-file /path/to/instances_train.json \
  --val-images-dir /path/to/val/images \
  --val-ann-file /path/to/instances_val.json \
  --checkpoint-dir weights \
  --log-file training_log.txt \
  --device cuda --dino --amp
```

Defaults include 200 epochs, batch size 4, 100 queries, detector learning rate `1e-4`, backbone learning rate `1e-5`, and gradient clipping at `0.1`. Reduce the batch size if GPU memory is limited. `python main.py --help` lists available options.

Each epoch evaluates the EMA model and writes:

| Output | Contents |
| --- | --- |
| `weights/best_model.pth` | EMA model state dictionary when validation AP improves above the initial best of zero. |
| `weights/last_model.pth` | EMA model state dictionary from the latest epoch. |
| `training_log.txt` | Epoch, training loss, validation loss, and AP@[0.50:0.95]. |
| `evaluation_results_last.json` | Latest epoch's COCO-format predictions, with pixel `[x, y, width, height]` boxes. |

The log is overwritten at the start of a run; the latest evaluation JSON and checkpoint names are also reused. Use separate checkpoint directories and log filenames to preserve experiments. The evaluation JSON path is fixed at the repository root.

The existing `--checkpoint` flag loads the hardcoded file `weights/hist/aug_100ep.pth`. It only loads model weights; it does not restore the optimizer, epoch counter, or historical best AP. Leave it off for a new experiment, especially if that file is unavailable.

## 3. Inspect validation detections

After training has produced `evaluation_results_last.json`:

```bash
python vis_val.py --draw_gt --score_thr 0.05 --topk 30
```

Images are written to `vis/`, with ground truth in green and predictions in blue. Use `--image_id` to inspect a specific COCO image ID. Custom paths use underscore-style flags: `--images_dir`, `--ann_file`, `--det_json`, and `--out_dir`.

The JSON contains predictions from the latest evaluated epoch, which may differ from the best checkpoint.

## 4. Predict on new images

With a trained checkpoint available:

```bash
python inference.py \
  --images testdata/img \
  --weights weights/best_model.pth \
  --out vis_test \
  --device cuda \
  --topk 30
```

Use `weights/last_model.pth` if no best checkpoint was saved. CPU inference uses `--device cpu`. If training used a different query count, pass the same `--queries` value here; the checkpoint is loaded strictly and must match the model architecture.

Inference recursively searches for lowercase `*.jpg` files. It saves annotated images and `vis_test/predictions_raw.json`, whose entries contain `file_name`, pixel `bbox_xyxy` (`[x1, y1, x2, y2]`), `score`, and `category_id: 1`.

The active inference path keeps the highest-scoring queries without a confidence threshold or non-maximum suppression. `--topk 30` limits the number of predictions; it does not mean 30 pigs were confidently detected. The active path also does not clip boxes to image boundaries. If consuming predictions elsewhere, filter scores and clip coordinates as appropriate. Use unique image basenames across subfolders because saved outputs and the JSON use basenames only.

## 5. Export the submission CSV

When inference used the default output directory:

```bash
python make_submission.py
```

The script reads the hardcoded `vis_test/predictions_raw.json` and writes `submission.csv` with columns `Image_ID` and `PredictionString`. Each prediction contributes:

```text
confidence x y width height class
```

Coordinates are in pixels and the exported pig class is **0**. Multiple predictions are concatenated into one space-separated string per image. Numeric filename stems become image IDs; nonnumeric stems fall back to sequential IDs (a numeric zero also uses the fallback). Images with no prediction entries do not receive a CSV row. This is the project's submission format, so check the format and image-ID rules expected by your own evaluation system.

## Scope and limitations

The repository contains an experimental training pipeline, not a packaged application. Dataset preparation, dependency versions, random seeds, and the original run configuration are not recorded. Some comments in the scripts mention older filenames or settings; use the actual filenames and commands above. Data, annotations, and checkpoints must be supplied locally to train or run inference.

The `.gitignore` excludes local data, checkpoints, generated outputs, and future training logs. Git continues tracking files already committed, including the existing `training_log.txt`.
