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

class NumpyCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        # Stitches stored as numpy array: [x1, y1, x2, y2, r, g, b]
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.bounds = (0.0, 0.0, 0.0, 0.0)  # min_x, min_y, max_x, max_y
        self.zoom = 1.0
        self.pan_x, self.pan_y = 400.0, 300.0

        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def load_stitches(self, segs):
        if not segs:
            self.stitches_np = np.zeros((0, 7), dtype=np.float32)
            self.bounds = (0.0, 0.0, 0.0, 0.0)
            return

        self.stitches_np = np.array(segs, dtype=np.float32)

        # Calculate bounding box using NumPy vector operations
        min_x = float(np.min(self.stitches_np[:, [0, 2]]))
        max_x = float(np.max(self.stitches_np[:, [0, 2]]))
        min_y = float(np.min(self.stitches_np[:, [1, 3]]))
        max_y = float(np.max(self.stitches_np[:, [1, 3]]))
        self.bounds = (min_x, min_y, max_x, max_y)

        print(f"NumPy array shape: {self.stitches_np.shape}")
        print(f"Bounds: {min_x:.1f}, {min_y:.1f} to {max_x:.1f}, {max_y:.1f} mm")
        self.Refresh()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()

        # Iterate numpy rows directly
        for row in self.stitches_np:
            sx1 = int(row[0] * self.zoom + self.pan_x)
            sy1 = int(row[1] * self.zoom + self.pan_y)
            sx2 = int(row[2] * self.zoom + self.pan_x)
            sy2 = int(row[3] * self.zoom + self.pan_y)

            dc.SetPen(wx.Pen(wx.Colour(int(row[4]), int(row[5]), int(row[6])), 1))
            dc.DrawLine(sx1, sy1, sx2, sy2)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 6 (NumPy Array)", size=(1000, 700))
        self.canvas = NumpyCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = []
            lx, ly = 0.0, 0.0
            for st in pattern.stitches:
                x, y = st[0] / 10.0, st[1] / 10.0
                segs.append((lx, ly, x, y, 40, 120, 220))
                lx, ly = x, y
            self.canvas.load_stitches(segs)

        self.Centre()
        self.Show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args = parser.parse_args()

    app = wx.App()
    MainFrame(initial_file=args.pes_file)
    app.MainLoop()

