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

class ProgressBarTimeline(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent, size=(-1, 58))
        self.SetMinSize((-1, 58))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.color_boundaries = []
        self.visible_count = 0

        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def set_data(self, stitches_np, boundaries, visible_count):
        self.stitches_np = stitches_np
        self.color_boundaries = boundaries
        self.visible_count = visible_count
        self.Refresh()

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetClientSize()
        dc.SetBackground(wx.Brush(wx.Colour(235, 235, 235)))
        dc.Clear()

        total_stitches = self.stitches_np.shape[0]
        track_x = 15
        track_y = 12
        track_h = 16
        track_w = max(10, w - 2 * track_x)

        # Draw Color Timeline Segments
        if total_stitches > 0 and len(self.color_boundaries) > 0:
            bounds = self.color_boundaries + [total_stitches]
            for i in range(len(bounds) - 1):
                idx_start = bounds[i]
                idx_end = bounds[i + 1]
                if idx_start >= total_stitches: break

                # Fetch color from first stitch of segment
                r, g, b = int(self.stitches_np[idx_start, 4]), int(self.stitches_np[idx_start, 5]), int(self.stitches_np[idx_start, 6])

                x1 = track_x + int((idx_start / total_stitches) * track_w)
                x2 = track_x + int((idx_end / total_stitches) * track_w)
                seg_w = max(1, x2 - x1)

                dc.SetPen(wx.TRANSPARENT_PEN)
                dc.SetBrush(wx.Brush(wx.Colour(r, g, b)))
                dc.DrawRectangle(x1, track_y, seg_w, track_h)

        # Track Border
        dc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(track_x, track_y, track_w, track_h)

        # Draw Position Knob
        if total_stitches > 0:
            ratio = min(1.0, max(0.0, self.visible_count / total_stitches))
            knob_x = track_x + int(ratio * track_w)

            # Semi-transparent unstitched overlay
            dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0, 80)))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(knob_x, track_y, track_x + track_w - knob_x, track_h)

            # Indicator Pin
            dc.SetPen(wx.Pen(wx.Colour(220, 30, 30), 2))
            dc.DrawLine(knob_x, track_y - 3, knob_x, track_y + track_h + 3)

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="PES Viewer - Step 17 (Color Timeline ProgressBar)", size=(1000, 700))
        self.progress_bar = ProgressBarTimeline(self)

        # Mock stitch data with 3 colors
        segs = []
        for i in range(300): segs.append((0,0,1,1, 200, 40, 40))   # Red
        for i in range(400): segs.append((0,0,1,1, 40, 180, 40))   # Green
        for i in range(300): segs.append((0,0,1,1, 40, 40, 220))   # Blue

        stitches_np = np.array(segs, dtype=np.float32)
        self.progress_bar.set_data(stitches_np, [0, 300, 700], 450)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.Panel(self), 1, wx.EXPAND)
        sizer.Add(self.progress_bar, 0, wx.EXPAND)
        self.SetSizer(sizer)
        self.Centre()
        self.Show()

if __name__ == "__main__":
    app = wx.App()
    MainFrame()
    app.MainLoop()
