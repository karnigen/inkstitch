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

import argparse
import os
import wx

class ProgressBarPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent, size=(-1, 58))
        self.SetMinSize((-1, 58))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def OnSize(self, event):
        self.Refresh()
        event.Skip()

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetClientSize()

        # Draw base panel background
        dc.SetBackground(wx.Brush(wx.Colour(235, 235, 235)))
        dc.Clear()

        # Track bounds
        track_margin = 15
        track_y = 12
        track_h = 16
        track_w = max(10, w - 2 * track_margin)

        # Outer track frame
        dc.SetPen(wx.Pen(wx.Colour(180, 180, 180), 1))
        dc.SetBrush(wx.Brush(wx.Colour(210, 210, 210)))
        dc.DrawRectangle(track_margin, track_y, track_w, track_h)

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="PES Viewer - Step 16 (ProgressBar Base)", size=(1000, 700))
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Main Canvas Placeholder
        self.canvas_placeholder = wx.Panel(self)
        self.canvas_placeholder.SetBackgroundColour(wx.Colour(245, 245, 245))

        # Bottom Progress Panel
        self.progress_bar = ProgressBarPanel(self)

        sizer.Add(self.canvas_placeholder, 1, wx.EXPAND)
        sizer.Add(self.progress_bar, 0, wx.EXPAND)

        self.SetSizer(sizer)
        self.Centre()
        self.Show()

if __name__ == "__main__":
    app = wx.App()
    MainFrame()
    app.MainLoop()