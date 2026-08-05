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

class SeekingProgressBar(wx.Panel):
    def __init__(self, parent, on_seek_callback=None):
        super().__init__(parent, size=(-1, 58))
        self.SetMinSize((-1, 58))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.on_seek_callback = on_seek_callback
        self.total_stitches = 1000
        self.visible_count = 500
        self.is_dragging = False

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)

    def UpdateSeekFromMouse(self, mouse_x):
        w, _ = self.GetClientSize()
        track_x = 15
        track_w = max(10, w - 2 * track_x)

        ratio = (mouse_x - track_x) / float(track_w)
        ratio = max(0.0, min(1.0, ratio))

        new_count = int(ratio * self.total_stitches)
        self.visible_count = new_count
        self.Refresh()

        if self.on_seek_callback:
            self.on_seek_callback(new_count)

    def OnLeftDown(self, event):
        self.is_dragging = True
        self.CaptureMouse()
        self.UpdateSeekFromMouse(event.GetX())

    def OnLeftUp(self, event):
        if self.HasCapture():
            self.ReleaseMouse()
        self.is_dragging = False

    def OnMotion(self, event):
        if self.is_dragging and event.Dragging():
            self.UpdateSeekFromMouse(event.GetX())

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetClientSize()
        dc.SetBackground(wx.Brush(wx.Colour(235, 235, 235)))
        dc.Clear()

        track_x = 15
        track_y = 12
        track_h = 16
        track_w = max(10, w - 2 * track_x)

        dc.SetPen(wx.Pen(wx.Colour(120, 120, 120)))
        dc.SetBrush(wx.Brush(wx.Colour(200, 200, 200)))
        dc.DrawRectangle(track_x, track_y, track_w, track_h)

        # Position Knob
        ratio = self.visible_count / float(self.total_stitches)
        knob_x = track_x + int(ratio * track_w)
        dc.SetPen(wx.Pen(wx.Colour(200, 40, 40), 2))
        dc.DrawLine(knob_x, track_y - 2, knob_x, track_y + track_h + 2)

        dc.DrawText(f"Interactive Seeking (Drag Mouse): Stitch {self.visible_count} / {self.total_stitches}", 15, 34)

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="PES Viewer - Step 18 (Mouse Seeking)", size=(1000, 700))
        self.pb = SeekingProgressBar(self, on_seek_callback=self.OnSeek)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.Panel(self), 1, wx.EXPAND)
        sizer.Add(self.pb, 0, wx.EXPAND)
        self.SetSizer(sizer)
        self.Centre()
        self.Show()

    def OnSeek(self, count):
        print(f"Seek position updated to: {count}")

if __name__ == "__main__":
    app = wx.App()
    MainFrame()
    app.MainLoop()

