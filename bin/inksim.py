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
import pystitch as emb

class PesCanvasPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.stitches = []
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def set_stitches(self, stitches):
        self.stitches = stitches
        self.Refresh()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()

        if not self.stitches:
            dc.DrawText("No file loaded or empty file", 20, 20)
            return

        # Simple 1:1 render with hardcoded offset
        offset_x, offset_y = 200, 200
        for x1, y1, x2, y2, r, g, b in self.stitches:
            dc.SetPen(wx.Pen(wx.Colour(r, g, b), 1))
            dc.DrawLine(int(x1 + offset_x), int(y1 + offset_y), int(x2 + offset_x), int(y2 + offset_y))

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 4", size=(1000, 700))
        self.canvas = PesCanvasPanel(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = []
            lx, ly = 0, 0
            for st in pattern.stitches:
                x, y = st[0]/10.0, st[1]/10.0
                segs.append((lx, ly, x, y, 200, 50, 50))
                lx, ly = x, y
            self.canvas.set_stitches(segs)

        self.Centre()
        self.Show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args = parser.parse_args()

    app = wx.App()
    MainFrame(initial_file=args.pes_file)
    app.MainLoop()