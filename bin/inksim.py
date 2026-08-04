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

class BitmapCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.zoom = 1.0
        self.pan_x, self.pan_y = 400.0, 300.0

        self.cached_bitmap = None
        self.need_redraw = True

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def OnSize(self, event):
        self.need_redraw = True
        event.Skip()

    def render_to_buffer(self, w, h):
        # Create pure RGB pixel array initialized to white (255)
        buf = np.full((h, w, 3), 255, dtype=np.uint8)

        # Simple software line rasterization into numpy buffer
        for row in self.stitches_np:
            x1 = int(row[0] * self.zoom + self.pan_x)
            y1 = int(row[1] * self.zoom + self.pan_y)
            x2 = int(row[2] * self.zoom + self.pan_x)
            y2 = int(row[3] * self.zoom + self.pan_y)

            # Simple bounds check
            if 0 <= x1 < w and 0 <= y1 < h:
                buf[y1, x1] = [int(row[4]), int(row[5]), int(row[6])]
            if 0 <= x2 < w and 0 <= y2 < h:
                buf[y2, x2] = [int(row[4]), int(row[5]), int(row[6])]

        # Convert RGB numpy buffer to wx.Bitmap
        img = wx.Image(w, h)
        img.SetData(buf.tobytes())
        return wx.Bitmap(img)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)

        if not self.need_redraw and self.cached_bitmap:
            dc.DrawBitmap(self.cached_bitmap, 0, 0)
            return

        w, h = self.GetSize()
        if w < 1 or h < 1:
            return

        self.cached_bitmap = self.render_to_buffer(w, h)
        self.need_redraw = False
        dc.DrawBitmap(self.cached_bitmap, 0, 0)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 8 (Bitmap Buffer Rendering)", size=(1000, 700))
        self.canvas = BitmapCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = []
            lx, ly = 0.0, 0.0
            for st in pattern.stitches:
                x, y = st[0]/10.0, st[1]/10.0
                segs.append((lx, ly, x, y, 220, 80, 40))
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