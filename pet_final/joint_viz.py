"""Overlay wheat and pea predictions on the same base image from two eval output dirs.

Usage:
    python pet_final/joint_viz.py --wheat outputs_wheat_2048/preds --pea outputs_pea_2048/preds
    python pet_final/joint_viz.py --wheat outputs_wheat_1500/preds --pea outputs_pea_1500/preds

Requires points.json in each source dir (produced by eval_metrics.py, see eval.sh).
Output: pet_final/joint/<flat_name>/preds/ and gt/
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ORIG_IMG_DIR = ROOT / "annotations" / "images"

# BGR colors
WHEAT_PRED = (0, 0, 220)      # red
PEA_PRED   = (220, 0, 0)      # blue
WHEAT_GT   = (0, 123, 255)    # orange #ff7b00
PEA_GT     = (255, 68, 211)   # violet #d344ff


def _detect_res(path: Path) -> int:
    """Infer resize resolution from directory path (e.g. outputs_wheat_1500 → 1500), default 2048."""
    m = re.search(r"[_/](\d{3,4})(?:[_/]|$)", str(path))
    return int(m.group(1)) if m else 2048


def _detect_border_px(wheat: Path, pea: Path) -> float:
    """If 'bordure' appears in either source path, compute border_px."""
    if "bordure" in str(wheat) or "bordure" in str(pea):
        res = _detect_res(wheat)
        return 20.0 * res / 2048.0
    return 0.0


def _draw_circles(frame: np.ndarray, pts: np.ndarray, color: tuple, radius: int) -> None:
    for x, y in pts:
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1)


def _draw_border_rect(frame: np.ndarray, border_px: float) -> None:
    b = max(1, int(round(border_px)))
    H, W = frame.shape[:2]
    frame[:b, :] = 0
    frame[H - b:, :] = 0
    frame[:, :b] = 0
    frame[:, W - b:] = 0


def _add_legend(frame: np.ndarray, entries: list[tuple[str, tuple]]) -> np.ndarray:
    """Append a right-side legend strip with colored dot + label per entry."""
    H = frame.shape[0]
    legend_w = 190
    legend = np.full((H, legend_w, 3), 30, dtype=np.uint8)
    y = 30
    for label, color in entries:
        cv2.circle(legend, (18, y), 7, color, -1)
        cv2.putText(legend, label, (32, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)
        y += 38
    return np.concatenate([frame, legend], axis=1)


def _resize_img(img_path: Path, res: int) -> np.ndarray:
    """Open image and resize so long side = res; return BGR frame."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    scale = res / max(W, H)
    new_W, new_H = int(round(W * scale)), int(round(H * scale))
    img = img.resize((new_W, new_H), Image.BILINEAR)
    return np.array(img)[:, :, ::-1].copy()


def _flat_name(wheat: Path, pea: Path) -> str:
    """Produce a flat dir name by replacing path separators with underscores."""
    def _flatten(p: Path) -> str:
        # keep relative to pet_final/ parent if possible, else just str
        try:
            rel = p.relative_to(Path(__file__).parent)
        except ValueError:
            rel = p
        return str(rel).replace("/", "_").replace("\\", "_").strip("_")
    return f"{_flatten(wheat)}_{_flatten(pea)}"


def process(wheat_dir: Path, pea_dir: Path) -> None:
    res = _detect_res(wheat_dir)
    border_px = _detect_border_px(wheat_dir, pea_dir)

    # Load coordinate maps
    w_pts_path = wheat_dir / "points.json"
    p_pts_path = pea_dir  / "points.json"
    if not w_pts_path.exists():
        raise FileNotFoundError(f"points.json missing: {w_pts_path}\n"
                                "(rerun eval.sh to generate it)")
    if not p_pts_path.exists():
        raise FileNotFoundError(f"points.json missing: {p_pts_path}\n"
                                "(rerun eval.sh to generate it)")

    w_pts: dict = json.loads(w_pts_path.read_text())
    p_pts: dict = json.loads(p_pts_path.read_text())

    common_stems = sorted(set(w_pts) & set(p_pts))
    if not common_stems:
        print("No image is common to both directories.")
        return
    print(f"{len(common_stems)} common images. Resolution: {res} px. "
          f"Border: {border_px:.1f} px")

    out_base = Path(__file__).parent / "joint" / _flat_name(wheat_dir, pea_dir)
    preds_dir = out_base / "preds"
    gt_dir    = out_base / "gt"
    preds_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    joint_meta: dict = {}
    for stem in common_stems:
        orig_path = ORIG_IMG_DIR / f"{stem}.jpg"
        if not orig_path.exists():
            print(f"  Original image not found: {orig_path}, skipped")
            continue

        frame_base = _resize_img(orig_path, res)

        def _scaled(key: str, src: dict) -> np.ndarray:
            raw = src.get(stem, {}).get(key, [])
            if not raw:
                return np.zeros((0, 2))
            return np.array(raw, dtype=np.float32)

        w_pred = _scaled("pred_pts", w_pts)
        p_pred = _scaled("pred_pts", p_pts)
        w_gt   = _scaled("gt_pts",   w_pts)
        p_gt   = _scaled("gt_pts",   p_pts)

        joint_meta[stem] = {
            "wheat_pred": len(w_pred), "pea_pred": len(p_pred),
            "wheat_gt":   len(w_gt),   "pea_gt":   len(p_gt),
        }

        # preds/ — predicted points only
        frame = frame_base.copy()
        _draw_circles(frame, w_pred, WHEAT_PRED, radius=5)
        _draw_circles(frame, p_pred, PEA_PRED,   radius=5)
        if border_px > 0:
            _draw_border_rect(frame, border_px)
        frame = _add_legend(frame, [
            (f"Wheat pred ({len(w_pred)})", WHEAT_PRED),
            (f"Pea pred ({len(p_pred)})",   PEA_PRED),
        ])
        cv2.imwrite(str(preds_dir / f"{stem}.jpg"), frame)

        # gt/ — GT + predicted points
        frame = frame_base.copy()
        _draw_circles(frame, w_gt,   WHEAT_GT,   radius=8)
        _draw_circles(frame, p_gt,   PEA_GT,     radius=8)
        _draw_circles(frame, w_pred, WHEAT_PRED, radius=5)
        _draw_circles(frame, p_pred, PEA_PRED,   radius=5)
        if border_px > 0:
            _draw_border_rect(frame, border_px)
        frame = _add_legend(frame, [
            (f"Wheat pred ({len(w_pred)})", WHEAT_PRED),
            (f"Pea pred ({len(p_pred)})",   PEA_PRED),
            (f"Wheat GT ({len(w_gt)})",     WHEAT_GT),
            (f"Pea GT ({len(p_gt)})",       PEA_GT),
        ])
        cv2.imwrite(str(gt_dir / f"{stem}.jpg"), frame)

    # Write joint metadata for explore_outputs.py
    if joint_meta:
        meta_txt = json.dumps(joint_meta, indent=2)
        for d in (out_base, preds_dir, gt_dir):
            (d / "joint_results.json").write_text(meta_txt)

    print(f"Saved to {out_base}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheat", required=True,
                        help="Wheat results folder (e.g. pet_final/outputs_wheat_2048/preds)")
    parser.add_argument("--pea",   required=True,
                        help="Pea results folder (e.g. pet_final/outputs_pea_2048/preds)")
    args = parser.parse_args()
    process(Path(args.wheat), Path(args.pea))


if __name__ == "__main__":
    main()
