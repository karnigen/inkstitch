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
def render_grid_numba(buf, width, height, zoom, pan_x, pan_y, grid_step_mm=10.0):
    """Renders background millimeter grid directly into RGB numpy buffer."""
    # Background fill (light grey/white canvas: 245, 245, 245)
    buf[:, :, 0] = 245
    buf[:, :, 1] = 245
    buf[:, :, 2] = 245

    step_px = grid_step_mm * zoom
    if step_px < 5.0:  # Don't draw grid if lines are too dense
        return

    # Calculate visible world coordinates
    start_x_mm = -pan_x / zoom
    end_x_mm = (width - pan_x) / zoom
    start_y_mm = -pan_y / zoom
    end_y_mm = (height - pan_y) / zoom

    first_grid_x = int(np.floor(start_x_mm / grid_step_mm)) * grid_step_mm
    first_grid_y = int(np.floor(start_y_mm / grid_step_mm)) * grid_step_mm

    # Vertical grid lines
    curr_x = first_grid_x
    while curr_x <= end_x_mm:
        px = int(curr_x * zoom + pan_x)
        if 0 <= px < width:
            is_axis = abs(curr_x) < 1e-3
            color = 120 if is_axis else 220
            for y in range(height):
                buf[y, px, 0] = color
                buf[y, px, 1] = color
                buf[y, px, 2] = color
        curr_x += grid_step_mm

    # Horizontal grid lines
    curr_y = first_grid_y
    while curr_y <= end_y_mm:
        py = int(curr_y * zoom + pan_y)
        if 0 <= py < height:
            is_axis = abs(curr_y) < 1e-3
            color = 120 if is_axis else 220
            for x in range(width):
                buf[py, x, 0] = color
                buf[py, x, 1] = color
                buf[py, x, 2] = color
        curr_y += grid_step_mm

class GridCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.zoom = 5.0
        self.pan_x, self.pan_y = 500.0, 350.0

        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        w, h = self.GetSize()
        if w < 1 or h < 1:
            return

        buf = np.zeros((h, w, 3), dtype=np.uint8)
        render_grid_numba(buf, w, h, self.zoom, self.pan_x, self.pan_y)

        img = wx.Image(w, h)
        img.SetData(buf.tobytes())
        dc.DrawBitmap(wx.Bitmap(img), 0, 0)

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=f"PES Viewer - Step 10 (Fast Numba Grid)", size=(1000, 700))
        self.canvas = GridCanvas(self)
        self.Centre()
        self.Show()

if __name__ == "__main__":
    app = wx.App()
    MainFrame()
    app.MainLoop()