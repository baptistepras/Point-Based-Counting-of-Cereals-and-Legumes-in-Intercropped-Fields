"""Collect linear-meter transect corners via QGIS clicks.

Instructions:
  1. Open the orthomosaic in QGIS (project in EPSG:2154)
  2. Plugins -> Python Console -> Show Editor
  3. Paste this script into the editor and click Run
  4. RIGHT-click to record a corner — LEFT-click to pan/navigate
  5. linm_ref.json is updated in real time

Per-plot workflow:
  1. Call set_plot(X, Y)  — e.g. set_plot(79, 12)
  2. 4 right-clicks for 5SW: TL -> TR -> BR -> BL
  3. 4 right-clicks for 3NE: TL -> TR -> BR -> BL
  4. -> plot complete, call set_plot() again for the next one

Available commands:
  set_plot(X, Y)            -> start/resume capture for plot (X, Y)
  remove(X, Y)              -> delete both transects of (X, Y)
  remove(X, Y, "5SW")       -> delete only the 5SW transect
  remove(X, Y, "3NE")       -> delete only the 3NE transect
  undo()                    -> undo the last click (current session only)
  status()                  -> show progress
  stop()                    -> disable the click tool
  resume()                  -> re-enable it
  done()                    -> finish and print the final summary
"""

import json
from pathlib import Path
from qgis.gui import QgsMapTool, QgsMapToolPan
from qgis.core import (QgsCoordinateTransform, QgsCoordinateReferenceSystem,
                        QgsProject, QgsPointXY)
from qgis.PyQt.QtCore import Qt

# ── Config ─────────────────────────────────────────────────────────────────────
# Set this to the folder of this repository on your machine: the script is
# pasted into the QGIS console, so it cannot find its own location.
REPO_ROOT = Path.home() / "Point-Based-Counting-of-Cereals-and-Legumes-in-Intercropped-Fields"
OUTPUT_PATH = REPO_ROOT / "drone_extraction" / "linm_ref.json"
CORNERS  = ["TL", "TR", "BR", "BL"]
VARIANTS = ["5SW", "3NE"]   # always collect 5SW first, then 3NE

# ── Global state ──────────────────────────────────────────────────────────────

_linmeters: dict = {}
_current_plot: tuple[int, int] | None = None
_current_variant_idx: int = 0   # index into VARIANTS
_current_corner_idx:  int = 0   # index into CORNERS
_clicks: list[dict] = []        # session history, for undo()


def _key(x: int, y: int, variant: str) -> str:
    return f"{x}_{y}_{variant}"


def _parse_key(k: str) -> tuple[int, int, str]:
    """Return (x, y, variant) from a '{x}_{y}_{variant}' key."""
    xy_str, variant = k.rsplit("_", 1)
    x_str, y_str = xy_str.split("_")
    return int(x_str), int(y_str), variant


# ── Load existing file ────────────────────────────────────────────────────────

if OUTPUT_PATH.exists():
    try:
        data = json.loads(OUTPUT_PATH.read_text())
        _linmeters = data.get("linmeters", {})
        if _linmeters:
            n_complete = sum(1 for e in _linmeters.values() if len(e) == 4)
            print(f"Resuming: {n_complete}/{len(_linmeters)} complete transects loaded.")
    except Exception as e:
        print(f"Could not read {OUTPUT_PATH}: {e}")


# ── Save ───────────────────────────────────────────────────────────────────────

def _save() -> None:
    OUTPUT_PATH.write_text(json.dumps({"linmeters": _linmeters}, indent=2))


# ── Right-click callback ─────────────────────────────────────────────────────

_crs_l93 = QgsCoordinateReferenceSystem("EPSG:2154")


def _on_click(point: QgsPointXY) -> None:
    global _current_corner_idx, _current_variant_idx, _current_plot

    if _current_plot is None:
        print("No plot in progress. Call set_plot(X, Y) to start.")
        return

    crs_canvas = iface.mapCanvas().mapSettings().destinationCrs()
    if crs_canvas != _crs_l93:
        tf = QgsCoordinateTransform(crs_canvas, _crs_l93, QgsProject.instance())
        point = tf.transform(point)

    xy      = [round(point.x(), 4), round(point.y(), 4)]
    x, y    = _current_plot
    variant = VARIANTS[_current_variant_idx]
    corner  = CORNERS[_current_corner_idx]
    k       = _key(x, y, variant)

    if k not in _linmeters:
        _linmeters[k] = {}
    _linmeters[k][corner] = xy
    _clicks.append({"key": k, "corner": corner, "xy": xy})
    _save()

    _current_corner_idx += 1
    remaining_in_variant = 4 - _current_corner_idx

    if remaining_in_variant > 0:
        next_corner = CORNERS[_current_corner_idx]
        print(f"  {k}.{corner} OK   -> next: {next_corner} ({_current_corner_idx+1}/4)")
    else:
        print(f"  {k}.{corner} OK   -> transect {k} complete.")
        _current_corner_idx = 0
        _current_variant_idx += 1

        if _current_variant_idx < len(VARIANTS):
            next_variant = VARIANTS[_current_variant_idx]
            print(f"  Now: {next_variant} — TL (1/4)")
        else:
            print(f"Plot ({x}, {y}) complete! Call set_plot(X, Y) for the next one.")
            _current_plot        = None
            _current_variant_idx = 0


# ── Custom tool: left = pan, right = record ────────────────────────────────────

class _CollectorTool(QgsMapTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self._pan      = QgsMapToolPan(canvas)
        self._panning  = False

    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = True
            self._pan.canvasPressEvent(event)

    def canvasMoveEvent(self, event):
        if self._panning:
            self._pan.canvasMoveEvent(event)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = False
            self._pan.canvasReleaseEvent(event)
        elif event.button() == Qt.RightButton:
            _on_click(self.toMapCoordinates(event.pos()))

    def activate(self):
        super().activate()
        self.canvas().setCursor(Qt.CrossCursor)


_tool      = _CollectorTool(iface.mapCanvas())
_prev_tool = iface.mapCanvas().mapTool()


# ── Console commands ────────────────────────────────────────────────────────────

def set_plot(x: int, y: int) -> None:
    """Start or resume capture for plot (x, y)."""
    global _current_plot, _current_variant_idx, _current_corner_idx

    x, y = int(x), int(y)

    def _n_corners(variant: str) -> int:
        k = _key(x, y, variant)
        if k not in _linmeters:
            return 0
        return sum(1 for c in CORNERS if c in _linmeters[k])

    n_5sw = _n_corners("5SW")
    n_3ne = _n_corners("3NE")

    if n_5sw == 4 and n_3ne == 4:
        print(f"Plot ({x}, {y}) already complete (5SW + 3NE).")
        print(f"   To re-annotate: remove({x}, {y})  then set_plot({x}, {y})")
        return

    _current_plot = (x, y)

    # Decide which variant/corner to resume from
    if n_5sw == 4:
        # 5SW complete, start or resume 3NE
        _current_variant_idx = 1
        _current_corner_idx  = n_3ne
        if n_3ne == 0:
            print(f"Plot ({x}, {y}): 5SW already complete. Starting 3NE — TL (1/4)")
        else:
            print(f"Plot ({x}, {y}): 5SW complete, 3NE partial ({n_3ne}/4). "
                  f"Resuming at corner {CORNERS[n_3ne]}")
    elif n_3ne == 4:
        # 3NE complete, start or resume 5SW
        _current_variant_idx = 0
        _current_corner_idx  = n_5sw
        if n_5sw == 0:
            print(f"Plot ({x}, {y}): 3NE already complete. Starting 5SW — TL (1/4)")
        else:
            print(f"Plot ({x}, {y}): 3NE complete, 5SW partial ({n_5sw}/4). "
                  f"Resuming at corner {CORNERS[n_5sw]}")
    else:
        # Neither complete — start with 5SW
        _current_variant_idx = 0
        _current_corner_idx  = n_5sw
        if n_5sw == 0 and n_3ne == 0:
            print(f"Plot ({x}, {y}) — 5SW: TL (1/4)")
        else:
            print(f"Plot ({x}, {y}): 5SW {n_5sw}/4  3NE {n_3ne}/4. "
                  f"Resuming 5SW at corner {CORNERS[n_5sw]}")


def remove(x: int, y: int, variant: str | None = None) -> None:
    """Delete one or both transects of a plot from the JSON and the click history."""
    x, y = int(x), int(y)
    targets = [variant] if variant else VARIANTS
    removed = []
    for v in targets:
        k = _key(x, y, v)
        if k in _linmeters:
            del _linmeters[k]
            removed.append(k)
    global _clicks
    _clicks = [c for c in _clicks if c["key"] not in removed]
    if removed:
        _save()
        print(f"Removed: {', '.join(removed)}")
    else:
        targets_str = variant if variant else "5SW + 3NE"
        print(f"Nothing to remove for ({x}, {y}) [{targets_str}].")


def undo() -> None:
    """Undo the last recorded click in the current session."""
    global _current_plot, _current_variant_idx, _current_corner_idx

    if not _clicks:
        print("Nothing to undo (session history is empty).")
        return

    last   = _clicks.pop()
    k      = last["key"]
    corner = last["corner"]

    if k in _linmeters and corner in _linmeters[k]:
        del _linmeters[k][corner]
        if not _linmeters[k]:
            del _linmeters[k]
    _save()

    x, y, variant = _parse_key(k)
    _current_plot        = (x, y)
    _current_variant_idx = VARIANTS.index(variant)
    _current_corner_idx  = CORNERS.index(corner)

    print(f"Undone: {k}.{corner}   -> waiting for {k}.{CORNERS[_current_corner_idx]}")


def status() -> None:
    """Print current progress."""
    all_plots: set[tuple[int, int]] = set()
    for k in _linmeters:
        x, y, _ = _parse_key(k)
        all_plots.add((x, y))

    n_full = sum(
        1 for (x, y) in all_plots
        if all(_key(x, y, v) in _linmeters and len(_linmeters[_key(x, y, v)]) == 4
               for v in VARIANTS)
    )
    n_partial = len(all_plots) - n_full

    print(f"Transects: {len(_linmeters)} entries  |  "
          f"{n_full} complete plots  |  {n_partial} partial")

    if _current_plot:
        x, y    = _current_plot
        variant = VARIANTS[_current_variant_idx]
        corner  = CORNERS[_current_corner_idx]
        print(f"In progress: ({x}, {y}) -> {variant}.{corner} ({_current_corner_idx+1}/4)")
    else:
        print("No plot in progress. Call set_plot(X, Y) to start.")


def done() -> None:
    """Disable the tool and print the final summary."""
    stop()
    print()
    status()
    print(f"\nData saved to: {OUTPUT_PATH}")


def resume() -> None:
    """Re-enable the click-collection tool."""
    iface.mapCanvas().setMapTool(_tool)
    if _current_plot:
        x, y    = _current_plot
        variant = VARIANTS[_current_variant_idx]
        corner  = CORNERS[_current_corner_idx]
        print(f"Tool active. In progress: ({x}, {y}) -> {variant}.{corner}")
    else:
        print("Tool active. Call set_plot(X, Y) to start.")


def stop() -> None:
    """Disable the tool and hand control back to QGIS."""
    if _prev_tool:
        iface.mapCanvas().setMapTool(_prev_tool)
    print("Tool disabled. Call resume() to continue.")


# ── Startup ──────────────────────────────────────────────────────────────────

resume()
print()
print("Corner order: TL (top-left) -> TR (top-right) -> BR (bottom-right) -> BL (bottom-left)")
print("Per plot: 5SW (4 clicks) then 3NE (4 clicks)")
print("Commands: set_plot(X,Y)  remove(X,Y[,'5SW'|'3NE'])  undo()  status()  stop()  resume()  done()")
