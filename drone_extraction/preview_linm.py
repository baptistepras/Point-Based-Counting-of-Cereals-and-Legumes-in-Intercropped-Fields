"""Draw linm_ref.json's transect rectangles over the orthomosaic (sanity check before extracting).

Usage (from the MPAC root, i.e. the parent of wpcount/):
    python wpcount/drone_extraction/preview_linm.py

Reads wpcount/drone_extraction/linm_ref.json and writes
wpcount/drone_extraction/linm_preview.png. Adjust corners in linm_ref.json
(re-run qgis_click_collector.py) and re-run this to check the fit.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

HERE = Path(__file__).parent
TIF  = HERE.parent.parent / (
    "data/transfer_12651495_files_66ad680e/"
    "2025_12_22_vol_01_P4_RGB_STICSMIX_2_5m_45degrees_Metashape_001_orthomosaic_geo/"
    "2025_12_22_vol_01_P4_RGB_2_3_4_Metashape_001_orthomosaic_geo.tif"
)
LINM = HERE / "linm_ref.json"
OUT  = HERE / "linm_preview.png"
PX_W = 2500

# Colors per variant
COLORS = {"5SW": "cyan", "3NE": "orange"}


def main() -> None:
    if not LINM.exists():
        print(f"ERROR: {LINM} not found"); sys.exit(1)
    if not TIF.exists():
        print(f"ERROR: {TIF} not found"); sys.exit(1)

    linmeters = json.loads(LINM.read_text()).get("linmeters", {})
    complete  = {k: v for k, v in linmeters.items()
                 if all(c in v for c in ("TL", "TR", "BR", "BL"))}

    if not complete:
        print("No complete transect in linm_ref.json.")
        sys.exit(0)

    # Read the image at low resolution
    ds     = gdal.Open(str(TIF))
    gt     = ds.GetGeoTransform()
    full_w = ds.RasterXSize
    scale  = PX_W / full_w
    prev_h = round(ds.RasterYSize * scale)

    img = np.zeros((prev_h, PX_W, 3), dtype=np.uint8)
    for b in range(3):
        raw = ds.GetRasterBand(b + 1).ReadRaster(
            0, 0, full_w, ds.RasterYSize, buf_xsize=PX_W, buf_ysize=prev_h)
        img[:, :, b] = np.frombuffer(raw, dtype=np.uint8).reshape(prev_h, PX_W)
    ds = None

    def l93_to_px(x: float, y: float) -> tuple[float, float]:
        return (x - gt[0]) / gt[1] * scale, (y - gt[3]) / gt[5] * scale

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 14), dpi=150)
    ax.imshow(img, origin="upper", interpolation="none")

    for key, corners in complete.items():
        variant = key.rsplit("_", 1)[-1]
        color   = COLORS.get(variant, "red")
        pts     = [l93_to_px(*corners[c]) for c in ("TL", "TR", "BR", "BL")]
        poly    = mpatches.Polygon(pts, closed=True,
                                   edgecolor=color, facecolor="none",
                                   linewidth=1.0, alpha=0.9)
        ax.add_patch(poly)
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        ax.text(cx, cy, key, color="white", fontsize=3,
                ha="center", va="center", clip_on=True,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))

    # Legend
    legend = [mpatches.Patch(color=c, label=v) for v, c in COLORS.items()]
    ax.legend(handles=legend, loc="upper right", fontsize=7)

    ax.set_title(f"{len(complete)} transects — cyan=5SW  orange=3NE", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}")
    subprocess.run(["open", str(OUT)], check=False)


if __name__ == "__main__":
    main()
