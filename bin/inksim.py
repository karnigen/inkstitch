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

class FitCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.bounds = (0.0, 0.0, 0.0, 0.0)
        self.zoom = 1.0
        self.pan_x, self.pan_y = 0.0, 0.0

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.SetFocus()

    def FitToScreen(self):
        if self.stitches_np.shape[0] == 0:
            return

        min_x, min_y, max_x, max_y = self.bounds
        bw = max(1.0, max_x - min_x)
        bh = max(1.0, max_y - min_y)

        w, h = self.GetSize()
        if w < 10 or h < 10:
            w, h = 1000, 700

        zoom_x = (w * 0.8) / bw
        zoom_y = (h * 0.8) / bh
        self.zoom = min(zoom_x, zoom_y)

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        self.pan_x = w / 2.0 - cx * self.zoom
        self.pan_y = h / 2.0 - cy * self.zoom
        self.Refresh()

    def load_stitches(self, segs):
        if segs:
            self.stitches_np = np.array(segs, dtype=np.float32)
            self.bounds = (
                float(np.min(self.stitches_np[:, [0, 2]])),
                float(np.min(self.stitches_np[:, [1, 3]])),
                float(np.max(self.stitches_np[:, [0, 2]])),
                float(np.max(self.stitches_np[:, [1, 3]]))
            )
            wx.CallAfter(self.FitToScreen)

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        if key in (70, 102):  # 'F' or 'f'
            self.FitToScreen()
        else:
            event.Skip()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()
        for row in self.stitches_np:
            sx1 = int(row[0] * self.zoom + self.pan_x)
            sy1 = int(row[1] * self.zoom + self.pan_y)
            sx2 = int(row[2] * self.zoom + self.pan_x)
            sy2 = int(row[3] * self.zoom + self.pan_y)
            dc.SetPen(wx.Pen(wx.Colour(int(row[4]), int(row[5]), int(row[6])), 1))
            dc.DrawLine(sx1, sy1, sx2, sy2)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 7 (Fit To Screen - Press 'F')", size=(1000, 700))
        self.canvas = FitCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = [(0, 0, st[0]/10.0, st[1]/10.0, 30, 160, 30) for st in pattern.stitches]
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