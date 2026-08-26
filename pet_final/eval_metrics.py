"""Standalone PET evaluation reporting LOCA-style metrics.

Count per image = number of predicted foreground points.

Output (written under pet_final/outputs_[drone_]<species>_<resolution>[_bordure]/):
  preds/      - image + yellow predicted points
  preds_gt/   - image + blue GT points + yellow pred points
  preds_neg/  - neg-species images with yellow pred points (GT=0)
  vis/        - scatter + neg histogram
"""

from __future__ import annotations
from pathlib import Path
import sys
import math
import json
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PET"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util.misc as utils
from models import build_model
from datasets import build_dataset
from main import get_args_parser

ROOT = Path(__file__).resolve().parents[1]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_stats(pairs: list[tuple]) -> dict:
    """MAE, RMSE, MAPE, signed bias, R2 from (true, pred) pairs."""
    if not pairs:
        return {}
    n = len(pairs)
    errors = [p - t for t, p in pairs]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    rel = [(p - t) / t * 100 for t, p in pairs if t > 0]
    mape = sum(abs(r) for r in rel) / len(rel) if rel else None
    bias = sum(rel) / len(rel) if rel else None
    trues = [t for t, p in pairs]
    sum_gt = sum(trues)
    wmape = sum(abs(e) for e in errors) / sum_gt * 100 if sum_gt > 0 else None
    mean_t = sum_gt / n
    ss_tot = sum((t - mean_t) ** 2 for t in trues)
    ss_res = sum(e ** 2 for e in errors)
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    return {"n": n, "MAE": mae, "RMSE": rmse, "MAPE": mape, "wMAPE": wmape, "bias": bias, "R2": r2}


def compute_neg_stats(preds: list[int]) -> dict:
    """MAE and RMSE for negative test (GT=0 for all)."""
    if not preds:
        return {}
    n = len(preds)
    mae = sum(abs(p) for p in preds) / n
    rmse = math.sqrt(sum(p ** 2 for p in preds) / n)
    n_zero = sum(1 for p in preds if p == 0)
    return {"n": n, "MAE": mae, "RMSE": rmse, "n_zero": n_zero}


def print_stats(stats: dict, title: str) -> None:
    """Pretty-print the metric dict."""
    def f(v: float | None, suffix: str = "") -> str:
        return "n/a" if v is None else f"{v:.2f}{suffix}"
    print(f"\n=== {title} ===")
    print(f"n     : {stats.get('n', 0)}")
    print(f"MAE   : {f(stats.get('MAE'))}")
    print(f"RMSE  : {f(stats.get('RMSE'))}")
    if "MAPE" in stats:
        print(f"MAPE  : {f(stats.get('MAPE'), ' %')}")
        print(f"wMAPE : {f(stats.get('wMAPE'), ' %')}")
        print(f"bias  : {f(stats.get('bias'), ' %')}")
        print(f"R2    : {f(stats.get('R2'))}")
    if "n_zero" in stats:
        print(f"pred=0: {stats['n_zero']} / {stats['n']}")


# ── Point matching (TP/FP/FN) + border filtering ────────────────────────────
#
# Matching always runs first, on the full unfiltered point sets, and the
# border is applied afterward: a matched (TP) pair survives as long as at
# least one of its two points is outside the border — both points are kept
# and drawn even if one of them is geometrically inside the border band. This
# avoids the edge case where a real detection would otherwise be miscounted
# as a border-cut FP/FN purely because the border happened to fall between a
# GT point and its correctly-matched prediction. A pair is only dropped if
# BOTH its points are inside the border. Unmatched points (isolated FP/FN)
# are still filtered by the border individually, same as before.

def _in_border_mask(pts: np.ndarray, W: int, H: int, border_px: float) -> np.ndarray:
    """True for each point within border_px of any image edge."""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    b = border_px
    inside = (pts[:, 0] >= b) & (pts[:, 0] <= W - b) & (pts[:, 1] >= b) & (pts[:, 1] <= H - b)
    return ~inside


def _match_and_border(pred_pts: np.ndarray | None, gt_pts: np.ndarray | None, img_path: str,
                      threshold_px: float, border_px: float
                      ) -> tuple[np.ndarray, np.ndarray, int, int, int, float]:
    """Greedy-match the full point sets, then border-filter. See module note above.

    Returns (kept_pred_pts, kept_gt_pts, tp, fp, fn, tp_dist_sum).
    """
    pred_pts = pred_pts if pred_pts is not None else np.zeros((0, 2))
    gt_pts   = gt_pts   if gt_pts   is not None else np.zeros((0, 2))

    # Greedy distance-sorted matching on the full, unfiltered point sets.
    matched: dict[int, int] = {}
    if len(pred_pts) > 0 and len(gt_pts) > 0:
        diffs = pred_pts[:, None, :] - gt_pts[None, :, :]   # (P, G, 2)
        dists = np.sqrt((diffs ** 2).sum(-1))                # (P, G)
        pi, gi = np.where(dists <= threshold_px)
        if len(pi) > 0:
            d_vals = dists[pi, gi]
            order = np.argsort(d_vals)
            matched_g: set[int] = set()
            for idx in order:
                p, g = int(pi[idx]), int(gi[idx])
                if p not in matched and g not in matched_g:
                    matched[p] = g
                    matched_g.add(g)

    W = H = None
    if border_px > 0:
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            border_px = 0.0  # unreadable image — skip border filtering rather than crash

    pred_in_border = _in_border_mask(pred_pts, W, H, border_px) if border_px > 0 else np.zeros(len(pred_pts), dtype=bool)
    gt_in_border   = _in_border_mask(gt_pts,   W, H, border_px) if border_px > 0 else np.zeros(len(gt_pts),   dtype=bool)

    matched_gt_idx = set(matched.values())
    kept_pred_idx: list[int] = []
    kept_gt_idx:   list[int] = []
    tp = fp = fn = 0
    tp_dist_sum = 0.0

    for p, g in matched.items():
        if pred_in_border[p] and gt_in_border[g]:
            continue  # both ends inside the border — genuinely excluded
        kept_pred_idx.append(p)
        kept_gt_idx.append(g)
        tp += 1
        tp_dist_sum += float(np.linalg.norm(pred_pts[p] - gt_pts[g]))

    for p in range(len(pred_pts)):
        if p in matched or pred_in_border[p]:
            continue
        kept_pred_idx.append(p)
        fp += 1

    for g in range(len(gt_pts)):
        if g in matched_gt_idx or gt_in_border[g]:
            continue
        kept_gt_idx.append(g)
        fn += 1

    kept_pred = pred_pts[kept_pred_idx] if kept_pred_idx else np.zeros((0, 2))
    kept_gt   = gt_pts[kept_gt_idx]     if kept_gt_idx   else np.zeros((0, 2))
    return kept_pred, kept_gt, tp, fp, fn, tp_dist_sum


def _process_positive_rows(rows: list[tuple], threshold_px: float, border_px: float
                           ) -> tuple[list[tuple], int, int, int, float]:
    """Match + border-filter every same-species row. Returns (new_rows, tp, fp, fn, mean_tp_dist)."""
    new_rows: list[tuple] = []
    tp = fp = fn = 0
    tp_dist_sum = 0.0
    for gt, pred, stem, img_path, pred_pts, gt_pts in rows:
        kept_pred, kept_gt, t, f_p, f_n, d_sum = _match_and_border(
            pred_pts, gt_pts, img_path, threshold_px, border_px)
        tp += t; fp += f_p; fn += f_n
        tp_dist_sum += d_sum
        new_rows.append((len(kept_gt), len(kept_pred), stem, img_path, kept_pred, kept_gt))
    mean_dist = tp_dist_sum / tp if tp > 0 else 0.0
    return new_rows, tp, fp, fn, mean_dist


def _border_filter_points(pts: np.ndarray | None, img_path: str, border_px: float) -> np.ndarray:
    """Drop points within border_px of any edge — no matching involved.

    For negative/opposite-species rows, where the row's "gt_pts" is the other
    species' real annotations and matching predictions against them wouldn't
    be meaningful — only the predicted points matter there.
    """
    if pts is None or len(pts) == 0 or border_px <= 0:
        return pts if pts is not None else np.zeros((0, 2))
    try:
        with Image.open(img_path) as im:
            W, H = im.size
    except Exception:
        return pts
    b = border_px
    m = (pts[:, 0] >= b) & (pts[:, 0] <= W - b) & (pts[:, 1] >= b) & (pts[:, 1] <= H - b)
    return pts[m]


def _border_filter_neg_rows(rows: list[tuple], border_px: float) -> list[tuple]:
    """Border-filter predicted points only, for negative (opposite-species) rows."""
    if border_px <= 0:
        return rows
    filtered = []
    for gt, pred, stem, img_path, pred_pts, gt_pts in rows:
        new_pred = _border_filter_points(pred_pts, img_path, border_px)
        filtered.append((gt, len(new_pred), stem, img_path, new_pred, gt_pts))
    return filtered


def _print_detection_stats(tp: int, fp: int, fn: int, mean_dist: float, title: str,
                            threshold_px: float = 20.0) -> None:
    """Print TP/FP/FN + precision/recall/F1 + mean TP distance."""
    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:.1f} %"
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and (precision + recall) > 0 else None)
    print(f"\n--- {title} (threshold = {threshold_px:.0f} px) ---")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision   : {pct(precision)}")
    print(f"Recall      : {pct(recall)}")
    print(f"F1          : {pct(f1)}")
    print(f"Mean TP dist: {mean_dist:.1f} px")


# ── Visualizations ────────────────────────────────────────────────────────────

def _draw_circles(frame_bgr: np.ndarray, pts: np.ndarray,
                  color_bgr: tuple, radius: int = 5) -> None:
    """Draw filled circles in-place on a BGR frame."""
    for x, y in pts:
        cv2.circle(frame_bgr, (int(round(x)), int(round(y))), radius, color_bgr, -1)


def _draw_border_rect(frame_bgr: np.ndarray, border_px: float) -> None:
    """Paint black border bands on frame (in-place)."""
    b = max(1, int(round(border_px)))
    H, W = frame_bgr.shape[:2]
    frame_bgr[:b, :] = 0
    frame_bgr[H - b:, :] = 0
    frame_bgr[:, :b] = 0
    frame_bgr[:, W - b:] = 0


def _save_points_viz(out_dir: Path, stem: str, img_path: Path,
                     pred_pts: np.ndarray | None,
                     gt_pts: np.ndarray | None = None,
                     border_px: float = 0.0) -> None:
    """Save image with predicted (yellow, r=5) and optional GT (dark blue, r=8) points.

    Border rect is drawn *before* the points (not after) so a point kept despite
    sitting inside the border band (a matched pair split by the border — see
    _match_and_border) stays visible instead of being painted over.
    """
    img = Image.open(img_path).convert("RGB")
    frame = np.array(img)[:, :, ::-1].copy()
    n = len(pred_pts) if pred_pts is not None else 0
    if border_px > 0:
        _draw_border_rect(frame, border_px)
    if gt_pts is not None and len(gt_pts) > 0:
        _draw_circles(frame, gt_pts, color_bgr=(139, 0, 0), radius=8)
    if pred_pts is not None and n > 0:
        _draw_circles(frame, pred_pts, color_bgr=(0, 255, 255), radius=5)  # yellow BGR
    cv2.putText(frame, f"pred={n}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2)
    cv2.imwrite(str(out_dir / f"{stem}.jpg"), frame)


def _write_results_json(out_dir: Path, rows: list[tuple]) -> None:
    """Write {stem: {gt, pred}} to out_dir/results.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {stem: {"gt": int(gt), "pred": int(pred)} for gt, pred, stem, *_ in rows}
    (out_dir / "results.json").write_text(json.dumps(data, indent=2))


def _write_points_json(out_dir: Path, rows: list[tuple]) -> None:
    """Write {stem: {pred_pts, gt_pts}} to out_dir/points.json for joint visualization."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for row in rows:
        stem = row[2]
        pred_pts = row[4] if len(row) > 4 else None
        gt_pts   = row[5] if len(row) > 5 else None
        data[stem] = {
            "pred_pts": pred_pts.tolist() if pred_pts is not None and len(pred_pts) > 0 else [],
            "gt_pts":   gt_pts.tolist()   if gt_pts   is not None and len(gt_pts)   > 0 else [],
        }
    (out_dir / "points.json").write_text(json.dumps(data, indent=2))


# ── Scatter plot ──────────────────────────────────────────────────────────────

def _save_scatter(vis_dir: Path, tag: str,
                  pos_by_split: dict[str, list],
                  label: str = "PET") -> None:
    """Save positive scatter plot (pred vs GT) colored by split."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    split_colors = {"train": "#4477AA", "val": "#EE8833", "test": "#22AA44"}
    split_markers = {"train": "o", "val": "s", "test": "^"}
    fig, ax = plt.subplots(figsize=(7, 7))
    all_vals: list[float] = []
    for split, rows in pos_by_split.items():
        if not rows:
            continue
        gts   = [float(gt)   for gt, pred, *_ in rows]
        preds = [float(pred) for gt, pred, *_ in rows]
        all_vals.extend(gts + preds)
        ax.scatter(gts, preds,
                   c=split_colors.get(split, "gray"),
                   marker=split_markers.get(split, "o"),
                   s=40, alpha=0.75, label=f"{split} (n={len(rows)})", zorder=3)
    if all_vals:
        vmax = max(all_vals) * 1.05
        ax.plot([0, vmax], [0, vmax], "k--", lw=1, label="y = x")
        ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
    ax.set_xlabel("GT count"); ax.set_ylabel("Predicted count")
    ax.set_title(f"{label} {tag} — Positive (target-species images)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / f"scatter_{tag}_pos.png", dpi=120)
    plt.close(fig)
    print(f"Scatter plot → {vis_dir}/scatter_{tag}_pos.png")


def _save_neg_histogram(vis_dir: Path, tag: str,
                        neg_by_split: dict[str, list],
                        label: str = "PET") -> None:
    """Bar chart of pred counts for neg images (GT=0): X=pred count, Y=frequency."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    split_colors = {"train": "#4477AA", "val": "#EE8833", "test": "#22AA44"}
    all_preds = [pred for rows in neg_by_split.values() for _, pred, *_ in rows]
    if not all_preds:
        return
    max_val = max(all_preds)
    bins = np.arange(0, max_val + 2)
    fig, ax = plt.subplots(figsize=(9, 5))
    for split, rows in neg_by_split.items():
        preds = [pred for _, pred, *_ in rows]
        if not preds:
            continue
        ax.hist(preds, bins=bins, color=split_colors.get(split, "gray"),
                alpha=0.65, label=f"{split} (n={len(preds)})")
    ax.set_xlabel("Predicted count (error, GT=0 for all)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"{label} {tag} — Negative: error distribution")
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = vis_dir / f"neg_hist_{tag}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Neg histogram → {out}")


# ── Per-dataset inference ─────────────────────────────────────────────────────

def _infer_dataset(model, dataset, device, prefix: str = "") -> list[tuple]:
    """Infer on a dataset; return [(gt, pred, stem, img_path, pred_pts, gt_pts)]."""
    loader = DataLoader(dataset, 1, sampler=SequentialSampler(dataset),
                        drop_last=False, collate_fn=utils.collate_fn, num_workers=2)
    rows: list[tuple] = []
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            gt = targets[0]["points"].shape[0]
            img_path = targets[0].get("image_path", "")
            stem = Path(img_path).stem if img_path else "?"

            # GT points: (y, x) absolute px → convert to (x, y)
            gt_pts_yx = targets[0]["points"].cpu().numpy()
            gt_pts = gt_pts_yx[:, [1, 0]] if len(gt_pts_yx) > 0 else np.zeros((0, 2))

            outputs = model(samples, test=True, targets=targets)

            scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
            pred = len(scores)
            pred_pts: np.ndarray | None = None
            if "pred_points" in outputs:
                pts_norm = outputs["pred_points"][0].cpu().numpy()
                img_h_t, img_w_t = outputs["img_shape"]
                pred_pts = np.stack(
                    [pts_norm[:, 1] * img_w_t, pts_norm[:, 0] * img_h_t], axis=1)

            rows.append((gt, pred, stem, img_path, pred_pts, gt_pts))
            print(f"{prefix}{img_path}: gt={gt} pred={pred}")
    return rows


def _build_ds(image_set: str, root: Path, base_args) -> object | None:
    """Build a PET SHA dataset rooted at `root`; return None if unavailable."""
    if not root.exists():
        return None
    try:
        return build_dataset(image_set=image_set, args=base_args, data_root_override=str(root))
    except Exception:
        return None


# ── Output dir helpers ────────────────────────────────────────────────────────

def _make_variant_dirs(base: Path) -> dict[str, Path]:
    """Create and return the preds / preds_gt / preds_neg output dirs."""
    dirs = {
        "preds":     base / "preds",
        "preds_gt":  base / "preds_gt",
        "preds_neg": base / "preds_neg",
    }
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_pos_variants(dirs: dict, stem: str, img_path: str,
                       pred_pts: np.ndarray | None, gt_pts: np.ndarray,
                       border_px: float = 0.0) -> None:
    p = Path(img_path)
    if not p.exists():
        return
    _save_points_viz(dirs["preds"],    stem, p, pred_pts, border_px=border_px)
    _save_points_viz(dirs["preds_gt"], stem, p, pred_pts, gt_pts, border_px=border_px)


def _save_neg_variants(dirs: dict, stem: str, img_path: str,
                       pred_pts: np.ndarray | None,
                       border_px: float = 0.0) -> None:
    """Negative images: pred dots only (GT=0, no blue dots)."""
    p = Path(img_path)
    if not p.exists():
        return
    _save_points_viz(dirs["preds_neg"], stem, p, pred_pts, border_px=border_px)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Run PET inference and print LOCA-style metrics for one species/resolution."""
    import argparse as _ap
    # Parse our custom args first so --resolution is consumed before PET's parser sees it.
    # If PET's parser runs first, argparse prefix-matching maps --resolution -> --resume.
    _ep = _ap.ArgumentParser(add_help=False)
    _ep.add_argument("--pea", action="store_true")
    _ep.add_argument("--drone", action="store_true",
                     help="Evaluate against data_drone_<resolution>/ instead of data_<resolution>/")
    _ep.add_argument("--pred-json", type=str, default="")
    _ep.add_argument("--resolution", type=int, default=2048,
                     help="Long-side resolution used at data preparation (default 2048)")
    _ep.add_argument("--bordure", type=int, nargs="?", const=20, default=None,
                     help="Exclude points inside a border (px at 2048, scaled with --resolution);"
                          " no value = 20 px, absent = no border")
    _ep.add_argument("--px_wheat", type=int, default=20,
                     help="Wheat pred/GT matching threshold in px at 2048 (scaled with --resolution); default 20")
    _ep.add_argument("--px_pea", type=int, default=40,
                     help="Pea pred/GT matching threshold in px at 2048 (scaled with --resolution); default 40 "
                          "(pea tolerates a larger localization error than wheat before a match is wrong)")
    extra, remaining = _ep.parse_known_args()
    base_args = get_args_parser().parse_args(remaining)
    base_args.distributed = False

    res = extra.resolution
    _res_scale = res / 2048.0
    px_base = extra.px_pea if extra.pea else extra.px_wheat
    match_threshold = float(px_base) * _res_scale
    border_px: float = float(extra.bordure) * _res_scale if extra.bordure is not None else 0.0

    device = torch.device("cuda")

    is_pea = extra.pea
    species     = "pea" if is_pea else "wheat"
    opp_species = "wheat" if is_pea else "pea"
    spec  = "pea" if is_pea else "wheat"
    other = "wheat" if is_pea else "pea"

    # Shared data_<resolution>/ (or data_drone_<resolution>/) layout: {species}/
    # (train+val), {species}_test/ (test), and the opposite species' equivalents,
    # used for cross-species negative checks.
    _drone_part = "drone_" if extra.drone else ""
    data_root      = ROOT / "pet_final" / f"data_{_drone_part}{res}"
    train_root     = data_root / species
    opp_train_root = data_root / opp_species
    opp_test_root  = data_root / f"{opp_species}_test"

    _bordure_part = "_bordure" if border_px > 0 else ""
    out_root_base = Path(__file__).parent / f"outputs_{_drone_part}{species}_{res}{_bordure_part}"
    vis_dir = out_root_base / "vis"
    dirs = _make_variant_dirs(out_root_base)

    model, _ = build_model(base_args)
    model.to(device)
    ckpt = torch.load(base_args.resume, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Datasets. dataset_test relies on eval.sh's part_A -> data_<res>/{species}_test symlink.
    dataset_test  = build_dataset(image_set="val", args=base_args)
    dataset_train = _build_ds("train_eval", train_root, base_args)
    dataset_val   = _build_ds("val",        train_root, base_args)

    neg_dataset_train = _build_ds("train_eval", opp_train_root, base_args)
    neg_dataset_val   = _build_ds("val",        opp_train_root, base_args)
    neg_dataset_test  = _build_ds("val",        opp_test_root,  base_args)

    # Collect per-split results
    pos_by_split: dict[str, list] = {}
    neg_by_split: dict[str, list] = {}

    if dataset_train is not None:
        rows_train = _infer_dataset(model, dataset_train, device, prefix="[TRAIN] ")
        rows_train, tp, fp, fn, mean_dist = _process_positive_rows(rows_train, match_threshold, border_px)
        pos_by_split["train"] = rows_train
        print_stats(compute_stats([(gt, pred) for gt, pred, *_ in rows_train]),
                    f"PET {spec} — train")
        _print_detection_stats(tp, fp, fn, mean_dist, f"PET {spec} — train detection", threshold_px=match_threshold)
    else:
        pos_by_split["train"] = []

    # Neg-train independent of pos-train (decoupled to avoid silent failure cascade)
    if neg_dataset_train is not None:
        pos_train_stems = {s for _, _, s, *_ in pos_by_split.get("train", [])}
        neg_train_all = _infer_dataset(model, neg_dataset_train, device,
                                       prefix="[NEG-TRAIN] ")
        neg_train_all = _border_filter_neg_rows(neg_train_all, border_px)
        neg_train_excl = [(0, pred, s, ip, pp, gp)
                          for _, pred, s, ip, pp, gp in neg_train_all
                          if s not in pos_train_stems]
        neg_by_split["train"] = [(0, pred, s) for _, pred, s, *_ in neg_train_excl]
        if neg_train_excl:
            print_stats(compute_neg_stats([pred for _, pred, *_ in neg_train_excl]),
                        f"PET {spec} — Negative {other}-exclusive (train)")
    else:
        neg_by_split["train"] = []

    # Val split (user's held-out validation split used for checkpoint selection)
    if dataset_val is not None:
        rows_val = _infer_dataset(model, dataset_val, device, prefix="[VAL] ")
        rows_val, tp, fp, fn, mean_dist = _process_positive_rows(rows_val, match_threshold, border_px)
        pos_by_split["val"] = rows_val
        print_stats(compute_stats([(gt, pred) for gt, pred, *_ in rows_val]),
                    f"PET {spec} — val")
        _print_detection_stats(tp, fp, fn, mean_dist, f"PET {spec} — val detection", threshold_px=match_threshold)
        if neg_dataset_val is not None:
            pos_val_stems = {s for _, _, s, *_ in rows_val}
            neg_val_all = _infer_dataset(model, neg_dataset_val, device, prefix="[NEG-VAL] ")
            neg_val_all = _border_filter_neg_rows(neg_val_all, border_px)
            neg_val_excl = [(0, pred, s, ip, pp, gp) for _, pred, s, ip, pp, gp in neg_val_all
                            if s not in pos_val_stems]
            neg_by_split["val"] = [(0, pred, s) for _, pred, s, *_ in neg_val_excl]
            if neg_val_excl:
                print_stats(compute_neg_stats([pred for _, pred, *_ in neg_val_excl]),
                            f"PET {spec} — Negative {other}-exclusive (val)")
        else:
            neg_by_split["val"] = []

    # Test split
    named = _infer_dataset(model, dataset_test, device)

    named, tp, fp, fn, mean_dist = _process_positive_rows(named, match_threshold, border_px)
    pos_by_split["test"] = named
    print_stats(compute_stats([(gt, pred) for gt, pred, *_ in named]),
                f"PET {spec} — test")
    _print_detection_stats(tp, fp, fn, mean_dist, f"PET {spec} — test detection", threshold_px=match_threshold)

    # Negative evaluation
    neg_rows: list[tuple] = []
    if neg_dataset_test is not None:
        pos_stems = {s for _, _, s, *_ in named}
        neg_all = _infer_dataset(model, neg_dataset_test, device, prefix="[NEG] ")
        neg_all = _border_filter_neg_rows(neg_all, border_px)
        neg_rows = [(0, pred, s, ip, pp, gp) for _, pred, s, ip, pp, gp in neg_all
                    if s not in pos_stems]
        neg_by_split["test"] = [(0, pred, s) for _, pred, s, *_ in neg_rows]
        if neg_rows:
            print_stats(compute_neg_stats([pred for _, pred, *_ in neg_rows]),
                        f"PET {spec} — Negative {other}-exclusive (test)")
    else:
        neg_by_split["test"] = []

    if extra.pred_json:
        pred_dict = {stem: {"gt": gt, "pred": pred} for gt, pred, stem, *_ in named}
        Path(extra.pred_json).parent.mkdir(parents=True, exist_ok=True)
        Path(extra.pred_json).write_text(json.dumps(pred_dict, indent=2))
        print(f"Predictions saved: {extra.pred_json}")

    # ── Visualizations (test split) ───────────────────────────────────────────
    # Positive images
    for gt, pred, stem, img_path, pred_pts, gt_pts in named:
        _save_pos_variants(dirs, stem, img_path, pred_pts, gt_pts, border_px=border_px)
    _write_results_json(dirs["preds"],    named)
    _write_points_json(dirs["preds"],     named)
    _write_results_json(dirs["preds_gt"], named)
    _write_points_json(dirs["preds_gt"],  named)

    # Negative images
    for _, pred, stem, img_path, pred_pts, _ in neg_rows:
        _save_neg_variants(dirs, stem, img_path, pred_pts, border_px=border_px)
    neg_for_json = [(0, pred, s, ip, pp, gp) for _, pred, s, ip, pp, gp in neg_rows]
    _write_results_json(dirs["preds_neg"], neg_for_json)
    _write_points_json(dirs["preds_neg"],  neg_for_json)

    # ── Scatter plot ────────────────────────────────────────────────────────
    tag = f"{_drone_part}{species}_{res}"
    _save_scatter(vis_dir, tag, pos_by_split)
    _save_neg_histogram(vis_dir, tag, neg_by_split)


if __name__ == "__main__":
    main()
