"""Convert wheat + pea point annotations into PET's ShanghaiTech (SHA) layout.

Source layout (see --path, default wpcount/annotations/, or
wpcount/annotations_drone/ with --drone):
  <path>/images/                 all photos (wheat and pea annotations reference the same images)
  <path>/annotations_wheat.json
  <path>/annotations_pea.json

Output (written under pet_final/), one shared directory per resolution
(data_<resolution>/, or data_drone_<resolution>/ with --drone):
  data_<resolution>/wheat/       train split
  data_<resolution>/wheat_test/  test split
  data_<resolution>/pea/         train split
  data_<resolution>/pea_test/    test split
(each split root also contains train_data/ or test_data/, PET's SHA convention)

Points are stored as (x, y); PET's SHA loader flips them to (y, x) via
`['image_info'][0][0][0][0][0][:, ::-1]`. The exact MATLAB nesting the loader
expects is discovered by round-tripping candidates until one matches.
"""

from __future__ import annotations
from pathlib import Path
from io import BytesIO
import argparse
import json
import numpy as np
import scipy.io as sio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# split -> (dataset root, SHA subfolder) — wheat variant (pea built at runtime)
SPLIT_LAYOUT_WHEAT = {
    "train": ("wheat", "train_data"),
    "val":   ("wheat", "test_data"),
    "test":  ("wheat_test", "test_data"),
}
SPLIT_LAYOUT_PEA = {
    "train": ("pea", "train_data"),
    "val":   ("pea", "test_data"),
    "test":  ("pea_test", "test_data"),
}

_CHOSEN: list[tuple[bool, int]] = []


def resize_keep_long_side(
    img: Image.Image, pts: list[list[float]], long_side: int
) -> tuple[Image.Image, list[list[float]]]:
    """Resize img so its longest side = long_side; scale points accordingly."""
    W, H = img.size
    scale = long_side / max(W, H)
    if abs(scale - 1.0) < 1e-6:
        return img, pts
    nw, nh = round(W * scale), round(H * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    pts = [[x * scale, y * scale] for x, y in pts]
    return img, pts


def _make(loc: np.ndarray, use_struct: bool, depth: int) -> np.ndarray:
    """Wrap the (N, 2) point array in one candidate MATLAB struct/cell nesting."""
    base: np.ndarray
    if use_struct:
        base = np.zeros((1, 1), dtype=[("location", "O")])
        base[0, 0]["location"] = loc
    else:
        base = loc
    cur = base
    for _ in range(depth):
        wrap = np.zeros((1, 1), dtype=object)
        wrap[0, 0] = cur
        cur = wrap
    return cur


def _reads_back(image_info: np.ndarray, n: int) -> bool:
    """True if PET's exact SHA access expression recovers an (n, 2) array."""
    buf = BytesIO()
    sio.savemat(buf, {"image_info": image_info})
    buf.seek(0)
    m = sio.loadmat(buf)
    try:
        arr = m["image_info"][0][0][0][0][0]
    except (IndexError, TypeError):
        return False
    return isinstance(arr, np.ndarray) and arr.shape == (n, 2)


def build_image_info(points: np.ndarray) -> np.ndarray:
    """Build an image_info that matches PET's `['image_info'][0][0][0][0][0]` access."""
    loc = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if _CHOSEN:
        return _make(loc, *_CHOSEN[0])
    for use_struct in (True, False):
        for depth in range(5):
            cand = _make(loc, use_struct, depth)
            if _reads_back(cand, loc.shape[0]):
                _CHOSEN.append((use_struct, depth))
                print(f"MATLAB nesting: use_struct={use_struct}, depth={depth}")
                return cand
    raise RuntimeError("no MATLAB nesting matched PET's SHA loader")


def save_gt_mat(path: Path, points: np.ndarray) -> None:
    """Save a GT_*.mat and assert PET's loader recovers the (N, 2) point array."""
    n = np.asarray(points).reshape(-1, 2).shape[0]
    sio.savemat(str(path), {"image_info": build_image_info(points)})
    arr = sio.loadmat(str(path))["image_info"][0][0][0][0][0]
    assert arr.shape == (n, 2), f"{path}: got {arr.shape}, expected {(n, 2)}"


def prepare(
    ann_path: Path, img_dir: Path, out_dir: Path,
    pea: bool, long_side: int,
) -> None:
    """Build one species' PET SHA-style dataset inside out_dir, resized to long_side."""
    split_layout = SPLIT_LAYOUT_PEA if pea else SPLIT_LAYOUT_WHEAT
    ann = json.loads(ann_path.read_text())
    order = sorted(ann.items(), key=lambda kv: -len(kv[1]["points"]))
    counts: dict[str, list[int]] = {}

    for name, entry in order:
        split = entry["split"]
        if split not in split_layout:
            continue
        root, sub = split_layout[split]
        pts = [list(p) for p in entry["points"]]
        img = Image.open(img_dir / name).convert("RGB")
        img, pts = resize_keep_long_side(img, pts, long_side)
        pts_arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)

        d_img = out_dir / root / sub / "images"
        d_gt  = out_dir / root / sub / "ground-truth"
        d_img.mkdir(parents=True, exist_ok=True)
        d_gt.mkdir(parents=True, exist_ok=True)

        stem = Path(name).stem
        img.save(d_img / f"{stem}.jpg", quality=95)
        save_gt_mat(d_gt / f"GT_{stem}.mat", pts_arr)

        counts.setdefault(split, [0, 0])
        counts[split][0] += 1
        counts[split][1] += len(pts_arr)

    # Remove non-JPEG files (e.g. .DS_Store rsynced from macOS) that would break SHA.py's os.listdir
    for root, sub in split_layout.values():
        img_out = out_dir / root / sub / "images"
        if img_out.exists():
            for f in img_out.iterdir():
                if f.suffix.lower() != ".jpg":
                    f.unlink()
                    print(f"Removed: {f}")

    for split, (n_img, n_pts) in sorted(counts.items()):
        root, sub = split_layout[split]
        print(f"{split:5s}: {n_img:3d} images, {n_pts:6d} points -> {out_dir / root / sub}")


def main() -> None:
    """Entry point: prepare wheat + pea PET datasets for one or more resolutions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=None,
                        help="Folder with images/, annotations_wheat.json, annotations_pea.json "
                             "(default: wpcount/annotations/, or wpcount/annotations_drone/ with --drone)")
    parser.add_argument("--resolution", type=int, nargs="+", default=[2048],
                        help="Long-side target resolution(s) to prepare (default 2048)")
    parser.add_argument("--drone", action="store_true",
                        help="Use wpcount/annotations_drone/ as the default source and "
                             "write to data_drone_<resolution>/ instead of data_<resolution>/")
    args = parser.parse_args()

    default_ann_dir = "annotations_drone" if args.drone else "annotations"
    ann_root = args.path if args.path is not None else ROOT / default_ann_dir
    img_dir = ann_root / "images"
    base_dir = Path(__file__).resolve().parent
    data_prefix = "data_drone" if args.drone else "data"

    for res in args.resolution:
        out_dir = base_dir / f"{data_prefix}_{res}"
        print(f"=== Resolution {res} -> {out_dir.name}/ ===")
        prepare(ann_root / "annotations_wheat.json", img_dir, out_dir, pea=False, long_side=res)
        prepare(ann_root / "annotations_pea.json",   img_dir, out_dir, pea=True,  long_side=res)


if __name__ == "__main__":
    main()
