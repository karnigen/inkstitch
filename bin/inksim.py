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
def render_visible_stitches(buf, stitches, visible_count, width, height, zoom, pan_x, pan_y):
    """Renders only the first visible_count stitches from the array."""
    count = min(visible_count, stitches.shape[0])
    for i in range(count):
        st = stitches[i]
        sx0 = int(st[0] * zoom + pan_x)
        sy0 = int(st[1] * zoom + pan_y)
        sx1 = int(st[2] * zoom + pan_x)
        sy1 = int(st[3] * zoom + pan_y)

        # Frustum culling
        if (sx0 < 0 and sx1 < 0) or (sx0 >= width and sx1 >= width): continue
        if (sy0 < 0 and sy1 < 0) or (sy0 >= height and sy1 >= height): continue

        r, g, b = int(st[4]), int(st[5]), int(st[6])
        # Simple line drawing for step 12
        if 0 <= sx1 < width and 0 <= sy1 < height:
            buf[sy1, sx1] = [r, g, b]

class StepCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.visible_count = 0
        self.zoom = 2.0
        self.pan_x, self.pan_y = 400.0, 300.0

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.SetFocus()

    def set_visible_count(self, count):
        total = self.stitches_np.shape[0]
        self.visible_count = max(0, min(total, count))
        self.Refresh()

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        total = self.stitches_np.shape[0]

        if key == wx.WXK_RIGHT:
            self.set_visible_count(self.visible_count + 1)
        elif key == wx.WXK_LEFT:
            self.set_visible_count(self.visible_count - 1)
        elif key == wx.WXK_UP:
            self.set_visible_count(self.visible_count + 100)
        elif key == wx.WXK_DOWN:
            self.set_visible_count(self.visible_count - 100)
        elif key == wx.WXK_HOME:
            self.set_visible_count(0)
        elif key == wx.WXK_END:
            self.set_visible_count(total)
        else:
            event.Skip()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        w, h = self.GetSize()
        if w < 1 or h < 1: return

        buf = np.full((h, w, 3), 245, dtype=np.uint8)
        if self.stitches_np.shape[0] > 0:
            render_visible_stitches(buf, self.stitches_np, self.visible_count, w, h, self.zoom, self.pan_x, self.pan_y)

        img = wx.Image(w, h)
        img.SetData(buf.tobytes())
        dc.DrawBitmap(wx.Bitmap(img), 0, 0)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 12 (Visible Count Controls)", size=(1000, 700))
        self.canvas = StepCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = []
            lx, ly = 0.0, 0.0
            for st in pattern.stitches:
                x, y = st[0]/10.0, st[1]/10.0
                segs.append((lx, ly, x, y, 0, 150, 200))
                lx, ly = x, y
            self.canvas.stitches_np = np.array(segs, dtype=np.float32)
            self.canvas.visible_count = len(segs)

        self.Centre()
        self.Show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args = parser.parse_args()

    app = wx.App()
    MainFrame(initial_file=args.pes_file)
    app.MainLoop()