"""Pure PET inference on a folder of images (no dataset structure required).

Species and resolution are read from the checkpoint's path, which must follow
the pet_<wheat|pea>_<resolution> convention produced by train.sh (e.g.
PET/outputs/SHA/pet_wheat_1500/best_checkpoint.pth).

Saves visualizations to pet_final/inference/infer<TIMESTAMP>/
  normal/     - images with colored prediction dots
  bordures/   - same but border-filtered points + black border rectangle

Usage (via infer.sh or directly):
    python pet_final/infer.py --images /path/to/images --ckpt PET/outputs/SHA/pet_wheat_2048/best_checkpoint.pth
    python pet_final/infer.py --images /path/to/images --ckpt PET/outputs/SHA/pet_pea_1500/best_checkpoint.pth
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PET"))

from main import get_args_parser
from models import build_model
from util.misc import NestedTensor

# BGR colors matching joint_viz.py
WHEAT_COLOR = (0, 0, 220)   # red
PEA_COLOR   = (220, 0, 0)   # blue


def _draw_circles(frame: np.ndarray, pts: np.ndarray, color: tuple, radius: int = 5) -> None:
    for x, y in pts:
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1)


def _draw_border_rect(frame: np.ndarray, border_px: float) -> None:
    b = max(1, int(round(border_px)))
    H, W = frame.shape[:2]
    frame[:b, :] = 0
    frame[H - b:, :] = 0
    frame[:, :b] = 0
    frame[:, W - b:] = 0


def _filter_pts(pts: np.ndarray, W: int, H: int, b: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    m = (pts[:, 0] >= b) & (pts[:, 0] <= W - b) & \
        (pts[:, 1] >= b) & (pts[:, 1] <= H - b)
    return pts[m]


def _preprocess(img_path: Path, res: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load, resize to long side = res, pad to PET window-transformer multiples, normalize.

    PET's windowed encoder requires feature H divisible by 16 and W by 32 (backbone stride 8),
    so image H must be a multiple of 128 and W of 256. Padding is added bottom/right with
    zeros; the returned boolean mask marks padded positions as True (NestedTensor convention).
    Returns (tensor CHW padded, mask HW, H_orig, W_orig).
    """
    img = Image.open(img_path).convert("RGB")
    W0, H0 = img.size
    scale = res / max(W0, H0)
    new_W, new_H = int(round(W0 * scale)), int(round(H0 * scale))
    img = img.resize((new_W, new_H), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # pad to multiples of (128, 256) = backbone_stride * enc_win_(h, w)
    H_pad = (128 - new_H % 128) % 128
    W_pad = (256 - new_W % 256) % 256
    if H_pad or W_pad:
        t = F.pad(t, (0, W_pad, 0, H_pad))  # (left, right, top, bottom)
    H_tot, W_tot = new_H + H_pad, new_W + W_pad
    mask = torch.zeros(H_tot, W_tot, dtype=torch.bool)
    if H_pad:
        mask[new_H:, :] = True
    if W_pad:
        mask[:, new_W:] = True
    return t, mask, new_H, new_W


def _parse_ckpt(ckpt_path: Path) -> tuple[str, int]:
    """Infer (species, resolution) from a checkpoint path (pet_[drone_]<wheat|pea>_<resolution> convention)."""
    m = re.search(r"pet_(?:drone_)?(wheat|pea)_(\d+)", str(ckpt_path))
    if not m:
        sys.exit(f"Could not infer species/resolution from the checkpoint path: {ckpt_path}\n"
                 f"  The parent folder must follow the pet_[drone_]<wheat|pea>_<resolution>/ convention"
                 f" (e.g. PET/outputs/SHA/pet_wheat_2048/best_checkpoint.pth"
                 f" or PET/outputs/SHA/pet_drone_wheat_2048/best_checkpoint.pth).")
    return m.group(1), int(m.group(2))


def _save_viz(out_dir: Path, stem: str, img_path: Path, pred_pts: np.ndarray,
              color: tuple, border_px: float = 0.0) -> None:
    img = Image.open(img_path).convert("RGB")
    frame = np.array(img)[:, :, ::-1].copy()
    pts_to_draw = pred_pts
    if border_px > 0:
        W, H = img.size
        pts_to_draw = _filter_pts(pred_pts, W, H, border_px)
    _draw_circles(frame, pts_to_draw, color)
    if border_px > 0:
        _draw_border_rect(frame, border_px)
    cv2.putText(frame, f"pred={len(pts_to_draw)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imwrite(str(out_dir / f"{stem}.jpg"), frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True,
                        help="Folder containing the images to process")
    parser.add_argument("--ckpt",   required=True,
                        help="Path to the PET checkpoint (.pth), inside a "
                             "pet_<wheat|pea>_<resolution>/ folder")
    args = parser.parse_args()

    images_dir = Path(args.images)
    ckpt_path = Path(args.ckpt)
    species, resolution = _parse_ckpt(ckpt_path)
    is_pea = species == "pea"

    _IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    img_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in _IMG_SUFFIXES
    ) if images_dir.exists() else []
    if not img_files:
        sample = [p.name for p in images_dir.iterdir()][:10] if images_dir.exists() else []
        sys.exit(f"No image found in {images_dir}\n  (contents: {sample})")

    color     = PEA_COLOR if is_pea else WHEAT_COLOR
    border_px = 20.0 * resolution / 2048.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root     = Path(__file__).parent / "inference" / f"infer{ts}"
    normal_dir   = out_root / "normal"
    bordures_dir = out_root / "bordures"
    normal_dir.mkdir(parents=True, exist_ok=True)
    bordures_dir.mkdir(parents=True, exist_ok=True)

    # Build model with default PET args
    pet_args = get_args_parser().parse_args(["--dataset_file", "SHA",
                                             "--resume", str(ckpt_path)])
    pet_args.distributed = False
    model, _ = build_model(pet_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print(f"Model loaded from {ckpt_path}")
    print(f"{'Pea' if is_pea else 'Wheat'}  |  resolution={resolution}  |  border={border_px:.1f}px")
    print(f"Output -> {out_root}/")

    preds_log:        dict = {}
    preds_border_log: dict = {}

    with torch.no_grad():
        for img_path in img_files:
            stem = img_path.stem
            try:
                tensor, mask_2d, H, W = _preprocess(img_path, resolution)
            except Exception as e:
                print(f"  Error loading {img_path.name}: {e}")
                continue

            tensors = tensor.unsqueeze(0).to(device)
            mask    = mask_2d.unsqueeze(0).to(device)
            samples = NestedTensor(tensors, mask)

            outputs = model(samples, test=True)

            pts_norm = outputs["pred_points"][0].cpu().numpy()
            img_h_t, img_w_t = outputs["img_shape"]
            pred_pts = np.stack(
                [pts_norm[:, 1] * img_w_t, pts_norm[:, 0] * img_h_t], axis=1
            )

            pts_b = _filter_pts(pred_pts, int(img_w_t), int(img_h_t), border_px)

            preds_log[stem]        = len(pred_pts)
            preds_border_log[stem] = len(pts_b)
            print(f"  {img_path.name}: pred={len(pred_pts)}  bordure={len(pts_b)}")

            _save_viz(normal_dir,   stem, img_path, pred_pts, color, border_px=0.0)
            _save_viz(bordures_dir, stem, img_path, pred_pts, color, border_px=border_px)

    (normal_dir   / "preds.json").write_text(json.dumps(preds_log,        indent=2))
    (bordures_dir / "preds.json").write_text(json.dumps(preds_border_log, indent=2))

    print(f"\nDone. {len(img_files)} images -> {out_root}/")


if __name__ == "__main__":
    main()
