#!/usr/bin/env python3

from pathlib import Path
import sys
import os
import argparse

#-------------------------------------------------------------------
# Dual run: python3 or .venv python

script_dir = Path(__file__).resolve().parent
venv_dir = (script_dir / ".." / ".venv").resolve()

def ensure_venv():
    # print(f"{sys.prefix=}"); print(f"{sys.base_prefix=}")
    if sys.prefix != sys.base_prefix: # we are in a virtual environment
        return

    # Path to the Python .venv (Linux/macOS vs Windows)
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        print(f"Warning: Virtual environment not found at {venv_dir}. Running with system Python.", file=sys.stderr)

# start the virtual environment if not already active
ensure_venv()
#-------------------------------------------------------------------

import wx
import numpy as np
import pystitch as emb

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(fastmath=True)
def draw_line_shaded(buf, width, height, x0, y0, x1, y1, r, g, b, thickness=1):
    """Bresenham-based line rendering with 3D thread shading effect."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    total_steps = max(dx, dy)
    step = 0

    while True:
        if 0 <= x < width and 0 <= y < height:
            # Calculate subtle shading factor along line (thread highlight)
            factor = 1.0
            if total_steps > 0:
                progress = step / total_steps
                factor = 0.85 + 0.3 * np.sin(progress * np.pi)  # Lighter in middle

            cr = min(255, int(r * factor))
            cg = min(255, int(g * factor))
            cb = min(255, int(b * factor))

            buf[y, x, 0] = cr
            buf[y, x, 1] = cg
            buf[y, x, 2] = cb

            # Apply thickness offset if requested
            if thickness > 1:
                if dx > dy:
                    if y + 1 < height: buf[y + 1, x] = [cr, cg, cb]
                else:
                    if x + 1 < width: buf[y, x + 1] = [cr, cg, cb]

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        step += 1

@njit(fastmath=True)
def render_shaded_numba(buf, stitches, width, height, zoom, pan_x, pan_y, line_width=1):
    """Iterates all stitches, performs frustum culling, and renders shaded lines."""
    n_stitches = stitches.shape[0]
    for i in range(n_stitches):
        st = stitches[i]

        # Transform world coordinates to screen space
        sx0 = int(st[0] * zoom + pan_x)
        sy0 = int(st[1] * zoom + pan_y)
        sx1 = int(st[2] * zoom + pan_x)
        sy1 = int(st[3] * zoom + pan_y)

        # Frustum culling (skip line if completely outside screen)
        if (sx0 < 0 and sx1 < 0) or (sx0 >= width and sx1 >= width):
            continue
        if (sy0 < 0 and sy1 < 0) or (sy0 >= height and sy1 >= height):
            continue

        r, g, b = int(st[4]), int(st[5]), int(st[6])
        draw_line_shaded(buf, width, height, sx0, sy0, sx1, sy1, r, g, b, line_width)

class FastShadedCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.zoom = 2.5
        self.pan_x, self.pan_y = 400.0, 300.0

        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        w, h = self.GetSize()
        if w < 1 or h < 1:
            return

        buf = np.full((h, w, 3), 245, dtype=np.uint8)

        if self.stitches_np.shape[0] > 0:
            render_shaded_numba(buf, self.stitches_np, w, h, self.zoom, self.pan_x, self.pan_y, line_width=1)

        img = wx.Image(w, h)
        img.SetData(buf.tobytes())
        dc.DrawBitmap(wx.Bitmap(img), 0, 0)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 11 (Numba Shaded Stitches)", size=(1000, 700))
        self.canvas = FastShadedCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = []
            lx, ly = 0.0, 0.0
            for st in pattern.stitches:
                x, y = st[0]/10.0, st[1]/10.0
                segs.append((lx, ly, x, y, 180, 40, 90))
                lx, ly = x, y
            self.canvas.stitches_np = np.array(segs, dtype=np.float32)

        self.Centre()
        self.Show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args = parser.parse_args()

    app = wx.App()
    MainFrame(initial_file=args.pes_file)
    app.MainLoop()