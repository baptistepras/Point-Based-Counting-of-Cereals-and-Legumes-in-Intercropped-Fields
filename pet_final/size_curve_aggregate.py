"""Aggregate a data-size learning-curve sweep (size_curve.sh) into one plot.

Reads best_mae from each size's best_checkpoint.pth
(PET/outputs/SHA/size_curve_[drone_]<species>_<resolution>_n<N>/) and writes
pet_final/size_curve/<species>_<resolution>[_drone]_<timestamp>/
    sizes.json               -- {n_train, best_val_mae} per size
    size_curve_<species>_<resolution>.png

Called automatically at the end of size_curve.sh — run it by hand only to
re-plot an already-finished sweep (e.g. after editing sizes.json by hand).

Usage:
    python pet_final/size_curve_aggregate.py --wheat --resolution 2048 \\
        --sizes 20 40 60 80 100 --timestamp 20260101_120000
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheat", action="store_true")
    parser.add_argument("--pea",   action="store_true")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--drone", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", required=True,
                        help="Training-set sizes swept by size_curve.sh, in the same order")
    parser.add_argument("--timestamp", required=True,
                        help="Timestamp tag from size_curve.sh, so the output folder name matches its log")
    args = parser.parse_args()

    if args.wheat and args.pea:
        sys.exit("Error: --wheat and --pea are mutually exclusive")
    if not args.wheat and not args.pea:
        sys.exit("Error: select a species with --wheat or --pea")

    species    = "pea" if args.pea else "wheat"
    drone_part = "drone_" if args.drone else ""

    points: list[tuple[int, float]] = []
    missing: list[int] = []
    for n in args.sizes:
        run_dir   = ROOT / "PET" / "outputs" / "SHA" / f"size_curve_{drone_part}{species}_{args.resolution}_n{n}"
        ckpt_path = run_dir / "best_checkpoint.pth"
        if not ckpt_path.exists():
            missing.append(n)
            continue
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        points.append((n, float(ckpt["best_mae"])))

    if missing:
        print(f"WARNING: no checkpoint found for sizes {missing} — skipped "
              f"(expected under PET/outputs/SHA/size_curve_{drone_part}{species}_{args.resolution}_n<N>/)")
    if not points:
        sys.exit("No checkpoints found for any requested size — nothing to plot")

    points.sort()
    out_dir = ROOT / "pet_final" / "size_curve" / f"{drone_part}{species}_{args.resolution}_{args.timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "sizes.json").write_text(json.dumps(
        {"species": species, "resolution": args.resolution, "drone": args.drone,
         "points": [{"n_train": n, "best_val_mae": mae} for n, mae in points]},
        indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns, maes = zip(*points)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, maes, color="#4477AA", marker="o")
    ax.set_xlabel("Training images")
    ax.set_ylabel("Best val MAE")
    ax.set_title(f"PET — data-size learning curve ({species}, res={args.resolution})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / f"size_curve_{drone_part}{species}_{args.resolution}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    print(f"Sizes   : {list(ns)}")
    print(f"Val MAE : {[round(m, 2) for m in maes]}")
    print(f"Plot -> {fig_path}")
    print(f"Data -> {out_dir / 'sizes.json'}")


if __name__ == "__main__":
    main()
