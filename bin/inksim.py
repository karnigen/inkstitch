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

class InteractiveCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.stitches = []
        self.zoom = 1.0
        self.pan_x = 400.0
        self.pan_y = 300.0

        self.drag_start = None
        self.pan_start = (0, 0)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()

        for x1, y1, x2, y2, r, g, b in self.stitches:
            sx1 = int(x1 * self.zoom + self.pan_x)
            sy1 = int(y1 * self.zoom + self.pan_y)
            sx2 = int(x2 * self.zoom + self.pan_x)
            sy2 = int(y2 * self.zoom + self.pan_y)

            dc.SetPen(wx.Pen(wx.Colour(r, g, b), 1))
            dc.DrawLine(sx1, sy1, sx2, sy2)

    def OnWheel(self, event):
        mx, my = event.GetPosition()
        old_zoom = self.zoom

        if event.GetWheelRotation() > 0:
            self.zoom *= 1.15
        else:
            self.zoom /= 1.15

        self.zoom = max(0.1, min(30.0, self.zoom))

        scale = self.zoom / old_zoom
        self.pan_x = mx - scale * (mx - self.pan_x)
        self.pan_y = my - scale * (my - self.pan_y)
        self.Refresh()

    def OnLeftDown(self, event):
        self.drag_start = event.GetPosition()
        self.pan_start = (self.pan_x, self.pan_y)
        self.CaptureMouse()

    def OnLeftUp(self, event):
        if self.HasCapture():
            self.ReleaseMouse()
        self.drag_start = None

    def OnMotion(self, event):
        if self.drag_start and event.Dragging() and event.LeftIsDown():
            pos = event.GetPosition()
            dx = pos[0] - self.drag_start[0]
            dy = pos[1] - self.drag_start[1]
            self.pan_x = self.pan_start[0] + dx
            self.pan_y = self.pan_start[1] + dy
            self.Refresh()

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 5 (Zoom & Pan)", size=(1000, 700))
        self.canvas = InteractiveCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = []
            lx, ly = 0, 0
            for st in pattern.stitches:
                x, y = st[0]/10.0, st[1]/10.0
                segs.append((lx, ly, x, y, 0, 100, 200))
                lx, ly = x, y
            self.canvas.stitches = segs

        self.Centre()
        self.Show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args = parser.parse_args()

    app = wx.App()
    MainFrame(initial_file=args.pes_file)
    app.MainLoop()