"""Extract linear-meter transects from the orthomosaic into per-transect GeoTIFFs.

Usage (from the MPAC root, i.e. the parent of wpcount/):
    python wpcount/drone_extraction/extract_linm.py \\
        --tif      data/.../geo.tif \\
        --angle    45

Reads corner coordinates from linm_ref.json (produced by qgis_click_collector.py)
and writes one warped GeoTIFF per transect straight into wpcount/annotations_drone/images/,
ready for wpcount/annotations_drone/annotate.py to pick up.

Output naming: <plot>_<variant>_<angle>.tif, e.g. 79_12_5SW_45.tif

linm_ref.json:
    {
      "linmeters": {
        "79_12_5SW": {"TL": [x,y], "TR": [x,y], "BR": [x,y], "BL": [x,y]},
        "79_12_3NE": {"TL": [x,y], "TR": [x,y], "BR": [x,y], "BL": [x,y]},
        ...
      }
    }
    Coordinates are in Lambert-93 (EPSG:2154, meters).
    Each entry is extracted as a single GeoTIFF (no further tiling).
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

from osgeo import gdal

gdal.UseExceptions()


class PlotCorners(NamedTuple):
    TL: tuple[float, float]
    TR: tuple[float, float]
    BR: tuple[float, float]
    BL: tuple[float, float]


def _write_cutline_geojson(corners: PlotCorners, path: str) -> None:
    """Write a GeoJSON polygon (EPSG:2154) for the GDAL Warp cutline."""
    pts = [list(corners.TL), list(corners.TR), list(corners.BR),
           list(corners.BL), list(corners.TL)]
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2154"}},
        "features": [{"type": "Feature", "properties": {},
                      "geometry": {"type": "Polygon", "coordinates": [pts]}}],
    }
    with open(path, "w") as f:
        json.dump(geojson, f)


def warp_tile(src: str, corners: PlotCorners, out: str, pixel_m: float) -> None:
    """Extract one region from the source GeoTIFF using a polygon cutline."""
    tmp = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
    tmp.close()
    _write_cutline_geojson(corners, tmp.name)
    try:
        gdal.Warp(
            out, src,
            format="GTiff",
            xRes=pixel_m, yRes=pixel_m,
            cutlineDSName=tmp.name,
            cropToCutline=True,
            resampleAlg=gdal.GRA_Bilinear,
            multithread=True,
            warpMemoryLimit=2000,
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
        )
    finally:
        os.unlink(tmp.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif",      required=True,
                        help="Source orthomosaic GeoTIFF")
    parser.add_argument("--linm",     default="wpcount/drone_extraction/linm_ref.json",
                        help="Transect corners JSON (from qgis_click_collector.py)")
    parser.add_argument("--out",      default="wpcount/annotations_drone/images",
                        help="Output folder — defaults straight into annotate.py's image source")
    parser.add_argument("--pixel_mm", type=float, default=2.0,
                        help="Output resolution in mm/pixel (default 2.0)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-extract transects that already exist in --out")
    parser.add_argument("--angle",    required=True, choices=["45", "90"],
                        help="Flight angle (45 or 90) — appended to each tile's filename")
    args = parser.parse_args()

    pixel_m  = args.pixel_mm / 1000.0
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    data      = json.loads(Path(args.linm).read_text())
    linmeters = data.get("linmeters", {})

    # Keep only complete entries (4 corners)
    complete = {k: v for k, v in linmeters.items()
                if all(c in v for c in ("TL", "TR", "BR", "BL"))}
    incomplete = len(linmeters) - len(complete)

    suffix = f"(+ {incomplete} incomplete, skipped)  " if incomplete else ""
    print(f"{len(complete)} complete transects  {suffix}|  {args.pixel_mm} mm/px")

    for i, (key, raw) in enumerate(complete.items(), 1):
        out_tif = out_root / f"{key}_{args.angle}.tif"
        if not args.force and out_tif.exists():
            print(f"[{i:02d}/{len(complete)}] {key}_{args.angle} ... already extracted, skipped")
            continue

        print(f"[{i:02d}/{len(complete)}] {key}", end=" ... ", flush=True)
        corners = PlotCorners(
            TL=tuple(raw["TL"]), TR=tuple(raw["TR"]),
            BR=tuple(raw["BR"]), BL=tuple(raw["BL"]),
        )
        warp_tile(args.tif, corners, str(out_tif), pixel_m)

        ds = gdal.Open(str(out_tif))
        print(f"{ds.RasterXSize}x{ds.RasterYSize} px")
        ds = None

    print(f"\nDone -> {out_root}/")


if __name__ == "__main__":
    main()
