"""Point annotation tool for wpcount — wheat (red) and pea (blue) on the same image.

Usage:
    python annotate.py                # annotate every image in ./images/
    python annotate.py --angle 45     # drone data only: filter by flight-angle suffix (_45/_90)

Controls:
    Write + click      -> place a point of the active species
    Erase + click       -> remove the nearest point of the active species
    Scroll / pinch      -> zoom (centered on the cursor)
    Middle/right drag   -> pan
    Left/Right arrow    -> previous/next image

Species (Wheat/Pea buttons) select which color new points are placed in, but
both species live on the same image at once — this is an intercropped-plot
dataset, so a single photo/tile can (and usually does) need both.

Loads and saves ./annotations_wheat.json + ./annotations_pea.json in place,
in the exact format prepare_pet_data.py expects:
    {"<image filename>": {"points": [[x, y], ...], "split": "train"|"val"|"test"}}
An image only gets an entry in a given species' JSON if it has at least one
point of that species — saving an image with zero wheat points removes it
from annotations_wheat.json (and likewise for pea).

A newly-annotated image gets `"split": "?"` (unknown) rather than one of
train/val/test — prepare_pet_data.py skips "?" entries, so annotating doesn't
require deciding the split up front. Closing the window prints, separately
for wheat and pea (they don't have the same number of annotated images), how
many are in each split (including "?"); use assign_split.py --wheat/--pea to
batch-assign the "?" ones of one species once you have enough.
"""

from __future__ import annotations
import argparse
import json
import sys
import tkinter as tk
from collections import Counter
from tkinter import messagebox
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk

HERE       = Path(__file__).resolve().parent
IMG_DIR    = HERE / "images"
WHEAT_FILE = HERE / "annotations_wheat.json"
PEA_FILE   = HERE / "annotations_pea.json"

COLOR_WHEAT = "#E5383B"   # red
COLOR_PEA   = "#3A86FF"   # blue
POINT_R     = 5
PANEL_W     = 260
MAX_ZOOM    = 10.0

IMG_EXTS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
            "*.tif", "*.TIF", "*.tiff", "*.TIFF")


def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


class Annotator:
    def __init__(self, root: tk.Tk, images: list[Path]) -> None:
        self.root   = root
        self.images = images
        self.idx    = 0

        self.wheat_data: dict = json.loads(WHEAT_FILE.read_text()) if WHEAT_FILE.exists() else {}
        self.pea_data:   dict = json.loads(PEA_FILE.read_text())   if PEA_FILE.exists()   else {}

        self.species: str = "wheat"   # active species: wheat | pea
        self.mode:    str = "write"   # write | erase

        self.wheat_points: list[list[float]] = []
        self.pea_points:   list[list[float]] = []
        self.split: str = "?"

        self.scale:  float = 1.0
        self.img_w:  int   = 1
        self.img_h:  int   = 1
        self.pil_img: Optional[Image.Image] = None
        self._photo  = None

        self.zoom:        float                     = 1.0
        self.pan_x:       float                     = 0.0
        self.pan_y:       float                     = 0.0
        self._pan_anchor: Optional[tuple[int, int]] = None

        self._dirty: bool = False

        self._build_ui()
        self._load_image()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("wpcount annotator")
        self.root.configure(bg="#1a1a1a")

        self.canvas = tk.Canvas(self.root, bg="#0d0d0d", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>",   self._on_click)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        try:
            self.canvas.bind("<Magnify>", self._on_magnify)
        except Exception:
            pass  # not available on all Tk builds
        # Button-2 = right-click on macOS trackpad; Button-3 = real right button
        for b in (2, 3):
            self.canvas.bind(f"<ButtonPress-{b}>",  self._on_pan_start)
            self.canvas.bind(f"<B{b}-Motion>",       self._on_pan_drag)
            self.canvas.bind(f"<ButtonRelease-{b}>", self._on_pan_end)

        self.root.bind("<Left>",  lambda _e: self._go_prev())
        self.root.bind("<Right>", lambda _e: self._go_next())

        panel = tk.Frame(self.root, bg="#1a1a1a", width=PANEL_W)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        panel.pack_propagate(False)

        def lbl(text="", size=10, fg="#cccccc", **kw) -> tk.Label:
            return tk.Label(panel, text=text, bg="#1a1a1a", fg=fg,
                            font=("Helvetica", size), **kw)

        def sep() -> None:
            tk.Frame(panel, bg="#3a3a3a", height=1).pack(fill=tk.X, pady=5)

        def row() -> tk.Frame:
            f = tk.Frame(panel, bg="#1a1a1a")
            f.pack(fill=tk.X, pady=(2, 4))
            return f

        def navbtn(text: str, bg: str, cmd) -> tk.Label:
            b = tk.Label(panel, text=text, font=("Helvetica", 10, "bold"),
                        bg=bg, fg="white", pady=7, cursor="hand2", relief=tk.FLAT)
            b.bind("<Button-1>", lambda _e: cmd())
            return b

        self.lbl_file = lbl(size=9, fg="#aaaaaa", anchor="w",
                            wraplength=PANEL_W - 10, justify=tk.LEFT)
        self.lbl_file.pack(fill=tk.X)
        self.lbl_idx = lbl(size=9, fg="#666666", anchor="w")
        self.lbl_idx.pack(fill=tk.X)
        sep()

        lbl("Mode", size=9, fg="#777777").pack(anchor="w")
        r1 = row()
        self.btn_write = self._tbtn(r1, "Write", lambda: self._set_mode("write"))
        self.btn_erase = self._tbtn(r1, "Erase", lambda: self._set_mode("erase"))
        self.btn_write.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_erase.pack(side=tk.LEFT, fill=tk.X, expand=True)

        lbl("Active species", size=9, fg="#777777").pack(anchor="w")
        r2 = row()
        self.btn_wheat = self._tbtn(r2, "Wheat", lambda: self._set_species("wheat"))
        self.btn_pea   = self._tbtn(r2, "Pea",   lambda: self._set_species("pea"))
        self.btn_wheat.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_pea.pack(  side=tk.LEFT, fill=tk.X, expand=True)

        lbl("Split", size=9, fg="#777777").pack(anchor="w")
        r3 = row()
        self.btn_train = self._tbtn(r3, "Train", lambda: self._set_split("train"))
        self.btn_val   = self._tbtn(r3, "Val",   lambda: self._set_split("val"))
        self.btn_test  = self._tbtn(r3, "Test",  lambda: self._set_split("test"))
        self.btn_train.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        self.btn_val.pack(  side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        self.btn_test.pack( side=tk.LEFT, fill=tk.X, expand=True)

        sep()
        self.lbl_stats = lbl(size=10, fg="#aaaaaa", justify=tk.LEFT, anchor="w")
        self.lbl_stats.pack(fill=tk.X)
        sep()

        self.btn_prev  = navbtn("◀ Previous",         "#2a2a6a", self._go_prev)
        self.btn_reset = navbtn("⟳ Reset image",      "#6a2a2a", self._reset)
        self.btn_save  = navbtn("💾 Save",             "#1a5a2a", self._save)
        self.btn_next  = navbtn("Next ▶",              "#2a5a2a", self._go_next)
        self.btn_skip  = navbtn("⤵ First unannotated", "#3a3a3a", self._go_first_unannotated)
        for b in (self.btn_prev, self.btn_reset, self.btn_save, self.btn_next, self.btn_skip):
            b.pack(fill=tk.X, pady=2)

    def _tbtn(self, parent: tk.Frame, text: str, cmd) -> tk.Label:
        b = tk.Label(parent, text=text, font=("Helvetica", 9, "bold"),
                    bg="#222222", fg="#666666", pady=6, padx=2,
                    cursor="hand2", relief=tk.FLAT)
        b.bind("<Button-1>", lambda _e: cmd())
        return b

    # ── State setters ─────────────────────────────────────────────────────────

    def _set_mode(self, m: str) -> None:
        self.mode = m
        self._refresh_panel()

    def _set_species(self, s: str) -> None:
        self.species = s
        self._refresh_panel()

    def _set_split(self, s: str) -> None:
        self.split = s
        self._dirty = True
        self._refresh_panel()

    @staticmethod
    def _apply_btn(btn: tk.Label, active: bool, active_bg: str, active_fg: str = "white") -> None:
        if active:
            btn.config(bg=active_bg, fg=active_fg)
        else:
            btn.config(bg="#222222", fg="#777777")

    def _refresh_panel(self) -> None:
        ab = self._apply_btn
        ab(self.btn_write, self.mode == "write", "#2e7a2e")
        ab(self.btn_erase, self.mode == "erase", "#8a2a2a")
        ab(self.btn_wheat, self.species == "wheat", "#a5232a")
        ab(self.btn_pea,   self.species == "pea",   "#2255aa")
        ab(self.btn_train, self.split == "train", "#2e6e2e")
        ab(self.btn_val,   self.split == "val",   "#2a4a8a")
        ab(self.btn_test,  self.split == "test",  "#7a501a")

        stats = (f"Wheat : {len(self.wheat_points)} pts\nPea   : {len(self.pea_points)} pts"
                f"\nSplit : {self.split}")
        if self._dirty:
            stats += "\n(unsaved changes)"
        self.lbl_stats.config(text=stats)

    # ── Image loading & rendering ──────────────────────────────────────────────

    def _load_image(self) -> None:
        path = self.images[self.idx]
        self.current_path = path
        name = path.name
        self._dirty = False
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0

        w_entry = self.wheat_data.get(name)
        p_entry = self.pea_data.get(name)
        self.wheat_points = [list(p) for p in w_entry["points"]] if w_entry else []
        self.pea_points   = [list(p) for p in p_entry["points"]] if p_entry else []
        w_split = (w_entry or {}).get("split")
        p_split = (p_entry or {}).get("split")
        self.split = w_split or p_split or "?"

        self.pil_img = Image.open(path).convert("RGB")
        self.img_w, self.img_h = self.pil_img.size

        screen_h    = max(self.root.winfo_screenheight() - 80, 600)
        self.scale  = screen_h / self.img_h
        self._vp_w  = max(self.root.winfo_screenwidth() - PANEL_W, 100)
        self._vp_h  = screen_h

        self.lbl_file.config(text=name)
        self.lbl_idx.config( text=f"Image {self.idx + 1} / {len(self.images)}")

        self._redraw()
        self._refresh_panel()

    def _redraw(self) -> None:
        s  = self.scale * self.zoom
        cw, ch = int(self.img_w * s), int(self.img_h * s)
        px, py = int(self.pan_x), int(self.pan_y)

        scaled      = self.pil_img.resize((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(scaled)
        self.canvas.delete("all")
        self.canvas.create_image(px, py, anchor=tk.NW, image=self._photo)

        r = POINT_R
        for x, y in self.wheat_points:
            cx, cy = x * s + px, y * s + py
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=COLOR_WHEAT, outline="white", width=1)
        for x, y in self.pea_points:
            cx, cy = x * s + px, y * s + py
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=COLOR_PEA, outline="white", width=1)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def _on_click(self, ev: tk.Event) -> None:
        s  = self.scale * self.zoom
        ix = max(0.0, min(float(self.img_w - 1), (ev.x - self.pan_x) / s))
        iy = max(0.0, min(float(self.img_h - 1), (ev.y - self.pan_y) / s))
        if self.mode == "write":
            self._add_point(ix, iy)
        else:
            self._erase_nearest(ix, iy)

    def _add_point(self, ix: float, iy: float) -> None:
        pts = self.wheat_points if self.species == "wheat" else self.pea_points
        pts.append([ix, iy])
        self._dirty = True
        self._redraw()
        self._refresh_panel()

    def _erase_nearest(self, ix: float, iy: float) -> None:
        pts = self.wheat_points if self.species == "wheat" else self.pea_points
        if not pts:
            return
        i = min(range(len(pts)), key=lambda k: dist2(ix, iy, pts[k][0], pts[k][1]))
        pts.pop(i)
        self._dirty = True
        self._redraw()
        self._refresh_panel()

    # ── Zoom & pan ────────────────────────────────────────────────────────────

    def _on_wheel(self, ev: tk.Event) -> None:
        """Zoom in/out on scroll wheel (real mouse or two-finger trackpad)."""
        if ev.delta == 0:
            return
        factor = 1.15 if ev.delta > 0 else (1.0 / 1.15)
        self._do_zoom(factor, ev.x, ev.y)

    def _on_magnify(self, ev: tk.Event) -> None:
        """Zoom in/out on macOS pinch gesture."""
        factor = 1.0 + ev.delta
        if factor > 0.05:
            self._do_zoom(factor, ev.x, ev.y)

    def _do_zoom(self, factor: float, cx: float, cy: float) -> None:
        """Apply zoom factor centered on canvas point (cx, cy)."""
        new_zoom   = max(1.0, min(MAX_ZOOM, self.zoom * factor))
        self.pan_x = cx - (cx - self.pan_x) * new_zoom / self.zoom
        self.pan_y = cy - (cy - self.pan_y) * new_zoom / self.zoom
        self.zoom  = new_zoom
        self._clamp_pan()
        self._redraw()

    def _clamp_pan(self) -> None:
        """Clamp pan so the image stays within the viewport."""
        vp_w = getattr(self, '_vp_w', int(self.img_w * self.scale))
        vp_h = getattr(self, '_vp_h', int(self.img_h * self.scale))
        zw = self.img_w * self.scale * self.zoom
        zh = self.img_h * self.scale * self.zoom
        self.pan_x = 0.0 if zw <= vp_w else max(vp_w - zw, min(0.0, self.pan_x))
        self.pan_y = 0.0 if zh <= vp_h else max(vp_h - zh, min(0.0, self.pan_y))

    def _on_pan_start(self, ev: tk.Event) -> None:
        self._pan_anchor = (ev.x, ev.y)
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, ev: tk.Event) -> None:
        if self._pan_anchor is None:
            return
        self.pan_x += ev.x - self._pan_anchor[0]
        self.pan_y += ev.y - self._pan_anchor[1]
        self._pan_anchor = (ev.x, ev.y)
        self._clamp_pan()
        self._redraw()

    def _on_pan_end(self, _ev: tk.Event) -> None:
        self._pan_anchor = None
        self.canvas.config(cursor="crosshair")

    # ── Serialization ─────────────────────────────────────────────────────────

    def _commit(self) -> None:
        """Write the current image's points into self.wheat_data/self.pea_data (in memory)."""
        name = self.current_path.name
        if self.wheat_points:
            self.wheat_data[name] = {"points": [list(p) for p in self.wheat_points], "split": self.split}
        else:
            self.wheat_data.pop(name, None)
        if self.pea_points:
            self.pea_data[name] = {"points": [list(p) for p in self.pea_points], "split": self.split}
        else:
            self.pea_data.pop(name, None)

    def _flush(self) -> None:
        WHEAT_FILE.write_text(json.dumps(self.wheat_data, indent=2))
        PEA_FILE.write_text(json.dumps(self.pea_data, indent=2))
        self._dirty = False

    def _save(self) -> None:
        self._commit()
        self._flush()
        self._refresh_panel()
        messagebox.showinfo("Saved", f"Saved to {WHEAT_FILE.name} + {PEA_FILE.name}")

    # ── Navigation ────────────────────────────────────────────────────────────

    def _maybe_save_before_nav(self) -> bool:
        """Ask to save unsaved edits before navigating away. Return True to proceed."""
        if not self._dirty:
            return True
        resp = messagebox.askyesnocancel(
            "Unsaved changes",
            f"Save {self.current_path.name} before continuing?",
        )
        if resp is None:
            return False
        if resp:
            self._commit()
            self._flush()
        return True

    def _go_prev(self) -> None:
        if self.idx == 0 or not self._maybe_save_before_nav():
            return
        self.idx -= 1
        self._load_image()

    def _go_next(self) -> None:
        if not self._maybe_save_before_nav():
            return
        if self.idx >= len(self.images) - 1:
            messagebox.showinfo("Done", "All images have been viewed.")
            return
        self.idx += 1
        self._load_image()

    def _go_first_unannotated(self) -> None:
        if not self._maybe_save_before_nav():
            return
        for i, path in enumerate(self.images):
            if path.name not in self.wheat_data and path.name not in self.pea_data:
                self.idx = i
                self._load_image()
                return
        messagebox.showinfo("All annotated", "Every image already has an annotation.")

    def _reset(self) -> None:
        """Discard unsaved edits on this image and reload its last-saved state."""
        if not messagebox.askyesno("Reset image", "Discard unsaved edits on this image?"):
            return
        self._load_image()

    @staticmethod
    def _split_counts(data: dict) -> Counter:
        """Count one species' annotated images per split."""
        return Counter(entry.get("split", "?") for entry in data.values())

    def _on_close(self) -> None:
        if self._dirty:
            resp = messagebox.askyesnocancel(
                "Unsaved changes",
                f"Save {self.current_path.name} before closing?",
            )
            if resp is None:
                return
            if resp:
                self._commit()
                self._flush()

        # Reported per species, not combined — wheat and pea don't have the same
        # number of annotated images, and assign_split.py operates per species too.
        wheat_counts = self._split_counts(self.wheat_data)
        pea_counts   = self._split_counts(self.pea_data)
        lines = [f"Wheat: {len(self.wheat_data)} annotated images"]
        for split in ("train", "val", "test", "?"):
            lines.append(f"  {split:<6}: {wheat_counts.get(split, 0)}")
        lines.append(f"Pea: {len(self.pea_data)} annotated images")
        for split in ("train", "val", "test", "?"):
            lines.append(f"  {split:<6}: {pea_counts.get(split, 0)}")
        if self.wheat_data or self.pea_data:
            messagebox.showinfo("Annotation summary", "\n".join(lines))
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--angle", choices=["45", "90"],
                        help="Drone data only: filter images by flight-angle suffix (_45/_90)")
    args = parser.parse_args()

    if not IMG_DIR.exists():
        sys.exit(f"Images folder not found: {IMG_DIR}")

    images: list[Path] = []
    for ext in IMG_EXTS:
        images.extend(IMG_DIR.glob(ext))
    images = sorted(set(images))

    if args.angle:
        images = [p for p in images if p.stem.endswith(f"_{args.angle}")]
        print(f"Angle filter {args.angle}: {len(images)} images")

    if not images:
        sys.exit(f"No image found in {IMG_DIR}")

    print(f"{len(images)} images to annotate")
    print(f"Annotations -> {WHEAT_FILE.name} + {PEA_FILE.name}")
    if WHEAT_FILE.exists() or PEA_FILE.exists():
        n_wheat = len(json.loads(WHEAT_FILE.read_text())) if WHEAT_FILE.exists() else 0
        n_pea   = len(json.loads(PEA_FILE.read_text()))   if PEA_FILE.exists()   else 0
        print(f"  ({n_wheat} wheat entries, {n_pea} pea entries already present)")

    root = tk.Tk()
    Annotator(root, images)
    root.mainloop()


if __name__ == "__main__":
    main()
