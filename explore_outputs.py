#!/usr/bin/env python3
"""Visual explorer for pet_final/ output images.

Opens a two-panel window:
  Left  - collapsible tree of pet_final/outputs_*/, inference/, joint/ subdirectories
  Right - image viewer with zoom/pan and circular left/right navigation

results.json / points.json / joint_results.json / preds.json are read as-is
(eval_metrics.py, joint_viz.py, and infer.py already write them at run time;
this tool only displays them, it does not regenerate them from logs).

Usage:
    python explore_outputs.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

ROOT = Path(__file__).resolve().parent

def _all_model_roots() -> list[tuple[str, Path]]:
    """Return all (label, dir) pairs under pet_final/: outputs_* dirs + inference + joint."""
    roots: list[tuple[str, Path]] = []

    pet_dir = ROOT / "pet_final"
    if pet_dir.exists():
        for d in sorted(pet_dir.iterdir()):
            if d.is_dir() and d.name.startswith("outputs"):
                roots.append((f"pet_final  {d.name}", d))
        for name in ("inference", "joint"):
            d = pet_dir / name
            if d.exists():
                roots.append((f"pet_final  {name}", d))

    return roots


# resolved at startup
MODEL_ROOTS: list[tuple[str, Path]] = []
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_MAX_ZOOM  = 10.0
TREE_W     = 260


def copy_to_clipboard(text: str) -> None:
    """Copy text to the system clipboard (macOS/Linux/Windows)."""
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    elif sys.platform == "win32":
        subprocess.run(["clip"], input=text.encode(), check=True)
    else:
        subprocess.run(["xclip", "-selection", "clipboard"],
                       input=text.encode(), check=False)


def images_in(directory: Path) -> list[Path]:
    """Return sorted image paths directly inside directory (non-recursive)."""
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


class OutputExplorer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Output Explorer")
        self.root.configure(bg="#1a1a1a")

        self.images:   list[Path]                   = []
        self.img_idx:  int                          = 0
        self.pil_img                                = None
        self._photo                                 = None
        self.zoom:     float                        = 1.0
        self.pan_x:    float                        = 0.0
        self.pan_y:    float                        = 0.0
        self._pan_anchor: Optional[tuple[int, int]] = None
        self.img_w:    int                          = 1
        self.img_h:    int                          = 1
        self.scale:    float                        = 1.0

        self._iid_to_path: dict[str, Path] = {}
        self._results: dict = {}       # stem → metadata from whichever JSON was found
        self._results_type: str = "none"  # "results" | "joint" | "preds" | "none"

        self._build_ui()
        self._populate_tree()
        self._draw_placeholder()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        left = tk.Frame(self.root, bg="#1a1a1a", width=TREE_W)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background="#1a1a1a", foreground="#cccccc",
                        fieldbackground="#1a1a1a", borderwidth=0,
                        rowheight=24, font=("Helvetica", 10))
        style.map("Dark.Treeview",
                  background=[("selected", "#2a4a6a")],
                  foreground=[("selected", "white")])

        self.tree = ttk.Treeview(left, style="Dark.Treeview",
                                  show="tree", selectmode="browse")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=(6, 0))
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        btn_frame = tk.Frame(left, bg="#1a1a1a")
        btn_frame.pack(fill=tk.X, padx=4, pady=6)

        def mkbtn(text: str, cmd) -> tk.Label:
            lb = tk.Label(btn_frame, text=text, bg="#2a2a2a", fg="#cccccc",
                          font=("Helvetica", 9, "bold"), pady=7,
                          cursor="hand2", relief=tk.FLAT)
            lb.bind("<Button-1>", lambda _e: cmd())
            return lb

        self.btn_copy_path = mkbtn("Copy path", self._copy_path)
        self.btn_copy_name = mkbtn("Copy name", self._copy_name)
        self.btn_copy_path.pack(fill=tk.X, pady=(0, 2))
        self.btn_copy_name.pack(fill=tk.X)

        right = tk.Frame(self.root, bg="#0d0d0d")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_counts = tk.Label(right, text="", bg="#111111", fg="#dddddd",
                                    font=("Helvetica", 13, "bold"), pady=6)
        self.lbl_counts.pack(fill=tk.X)

        self.canvas = tk.Canvas(right, bg="#0d0d0d", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        try:
            self.canvas.bind("<Magnify>", self._on_magnify)
        except Exception:
            pass
        for b in (2, 3):
            self.canvas.bind(f"<ButtonPress-{b}>",  self._on_pan_start)
            self.canvas.bind(f"<B{b}-Motion>",       self._on_pan_drag)
            self.canvas.bind(f"<ButtonRelease-{b}>", self._on_pan_end)
        self.root.bind("<Left>",  lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(1))

        nav = tk.Frame(right, bg="#111111")
        nav.pack(fill=tk.X)

        self.lbl_info = tk.Label(nav, text="", bg="#111111", fg="#777777",
                                  font=("Helvetica", 9), anchor="w", padx=8)
        self.lbl_info.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def navbtn(text: str, cmd) -> tk.Label:
            lb = tk.Label(nav, text=text, bg="#2a2a2a", fg="white",
                          font=("Helvetica", 13, "bold"),
                          padx=18, pady=5, cursor="hand2")
            lb.bind("<Button-1>", lambda _e: cmd())
            return lb

        navbtn("◀", lambda: self._step(-1)).pack(side=tk.LEFT, padx=(0, 2), pady=4)
        navbtn("▶", lambda: self._step( 1)).pack(side=tk.LEFT, padx=(0, 8), pady=4)

    # ── Tree ──────────────────────────────────────────────────────────────────

    def _populate_tree(self) -> None:
        for label, out_dir in MODEL_ROOTS:
            iid = self.tree.insert("", "end", text=f"  {label}", open=False)
            self._iid_to_path[iid] = out_dir
            if out_dir.exists():
                self._add_children(iid, out_dir)

    def _add_children(self, parent_iid: str, directory: Path) -> None:
        """Recursively add subdirectories to the tree."""
        try:
            subdirs = sorted(p for p in directory.iterdir() if p.is_dir())
        except (PermissionError, FileNotFoundError):
            return
        for sub in subdirs:
            iid = self.tree.insert(parent_iid, "end",
                                    text=f"  {sub.name}", open=False)
            self._iid_to_path[iid] = sub
            self._add_children(iid, sub)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_tree_select(self, _event: tk.Event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        path = self._iid_to_path.get(sel[0])
        if path is None or not path.exists():
            return
        imgs = images_in(path)
        if not imgs:
            return
        self.images  = imgs
        self.img_idx = 0
        joint_path   = path / "joint_results.json"
        results_path = path / "results.json"
        preds_path   = path / "preds.json"
        if joint_path.exists():
            self._results = json.loads(joint_path.read_text())
            self._results_type = "joint"
        elif results_path.exists():
            self._results = json.loads(results_path.read_text())
            self._results_type = "results"
        elif preds_path.exists():
            self._results = json.loads(preds_path.read_text())
            self._results_type = "preds"
        else:
            self._results = {}
            self._results_type = "none"
        self._load_image()

    # ── Image viewer ──────────────────────────────────────────────────────────

    def _load_image(self) -> None:
        from PIL import Image, ImageTk

        path = self.images[self.img_idx]
        self.pil_img = Image.open(path).convert("RGB")
        self.img_w, self.img_h = self.pil_img.size

        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 700
        self.scale = min(cw / self.img_w, ch / self.img_h)
        self.zoom  = 1.0
        self.pan_x = (cw - self.img_w * self.scale) / 2
        self.pan_y = (ch - self.img_h * self.scale) / 2

        self._redraw()
        rel = path.relative_to(ROOT)
        self.lbl_info.config(
            text=f"{self.img_idx + 1} / {len(self.images)}   {rel}")
        stem = path.stem
        meta = self._results.get(stem)
        if meta and self._results_type == "results":
            err = meta["pred"] - meta["gt"]
            sign = "+" if err >= 0 else ""
            self.lbl_counts.config(
                text=f"GT : {meta['gt']}     pred : {meta['pred']}     err : {sign}{err}")
        elif meta and self._results_type == "joint":
            wg = meta.get("wheat_gt", "?");  wp = meta.get("wheat_pred", "?")
            pg = meta.get("pea_gt",   "?");  pp = meta.get("pea_pred",   "?")
            self.lbl_counts.config(
                text=f"Wheat — GT : {wg}  pred : {wp}     Pea — GT : {pg}  pred : {pp}")
        elif self._results_type == "preds" and stem in self._results:
            self.lbl_counts.config(text=f"pred : {self._results[stem]}")
        else:
            self.lbl_counts.config(text="")

    def _redraw(self) -> None:
        from PIL import Image, ImageTk

        s  = self.scale * self.zoom
        cw = max(1, int(self.img_w * s))
        ch = max(1, int(self.img_h * s))

        scaled      = self.pil_img.resize((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(scaled)
        self.canvas.delete("all")
        self.canvas.create_image(int(self.pan_x), int(self.pan_y),
                                  anchor=tk.NW, image=self._photo)

    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            450, 350, text="Select a folder in the tree to view images",
            fill="#333333", font=("Helvetica", 14))
        self.lbl_counts.config(text="")

    def _step(self, delta: int) -> None:
        if not self.images:
            return
        self.img_idx = (self.img_idx + delta) % len(self.images)
        self._load_image()

    # ── Zoom & pan ────────────────────────────────────────────────────────────

    def _on_wheel(self, ev: tk.Event) -> None:
        if ev.delta == 0:
            return
        factor = 1.15 if ev.delta > 0 else (1.0 / 1.15)
        self._do_zoom(factor, ev.x, ev.y)

    def _on_magnify(self, ev: tk.Event) -> None:
        """macOS pinch-to-zoom gesture."""
        factor = 1.0 + ev.delta
        if factor > 0.05:
            self._do_zoom(factor, ev.x, ev.y)

    def _do_zoom(self, factor: float, cx: float, cy: float) -> None:
        """Zoom centered on canvas point (cx, cy)."""
        new_zoom = max(0.1, min(_MAX_ZOOM, self.zoom * factor))
        self.pan_x = cx - (cx - self.pan_x) * new_zoom / self.zoom
        self.pan_y = cy - (cy - self.pan_y) * new_zoom / self.zoom
        self.zoom  = new_zoom
        if self.pil_img:
            self._redraw()

    def _on_pan_start(self, ev: tk.Event) -> None:
        self._pan_anchor = (ev.x, ev.y)
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, ev: tk.Event) -> None:
        if self._pan_anchor is None:
            return
        self.pan_x += ev.x - self._pan_anchor[0]
        self.pan_y += ev.y - self._pan_anchor[1]
        self._pan_anchor = (ev.x, ev.y)
        if self.pil_img:
            self._redraw()

    def _on_pan_end(self, _ev: tk.Event) -> None:
        self._pan_anchor = None
        self.canvas.config(cursor="crosshair")

    # ── Clipboard ─────────────────────────────────────────────────────────────

    def _current_image(self) -> Optional[Path]:
        return self.images[self.img_idx] if self.images else None

    def _flash(self, btn: tk.Label, original: str) -> None:
        btn.config(text="Copied ✓", fg="#44cc44")
        self.root.after(1200, lambda: btn.config(text=original, fg="#cccccc"))

    def _copy_path(self) -> None:
        img = self._current_image()
        if img is None:
            return
        copy_to_clipboard(str(img.relative_to(ROOT)))
        self._flash(self.btn_copy_path, "Copy path")

    def _copy_name(self) -> None:
        img = self._current_image()
        if img is None:
            return
        copy_to_clipboard(img.name)
        self._flash(self.btn_copy_name, "Copy name")


def main() -> None:
    global MODEL_ROOTS
    MODEL_ROOTS = _all_model_roots()

    root = tk.Tk()
    root.geometry("1280x820")
    OutputExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
