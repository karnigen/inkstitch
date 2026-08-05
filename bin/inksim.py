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

import time
import wx
import numpy as np
import pystitch as emb

class ThrottledCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.visible_count = 0

        # Throttling parameters
        self._last_key_time = 0.0
        self._key_throttle_interval = 0.015  # Limit key events to max ~60 Hz

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.SetFocus()

    def OnKeyDown(self, event):
        now = time.time()
        # Suppress event if arriving faster than throttling interval
        if (now - self._last_key_time) < self._key_throttle_interval:
            return

        self._last_key_time = now
        key = event.GetKeyCode()
        total = self.stitches_np.shape[0]

        if key == wx.WXK_RIGHT:
            self.visible_count = min(total, self.visible_count + 1)
        elif key == wx.WXK_LEFT:
            self.visible_count = max(0, self.visible_count - 1)
        elif key == wx.WXK_UP:
            self.visible_count = min(total, self.visible_count + 100)
        elif key == wx.WXK_DOWN:
            self.visible_count = max(0, self.visible_count - 100)
        else:
            event.Skip()
            return

        self.Refresh()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()
        dc.DrawText(f"Throttled Input Active | Stitches: {self.visible_count} / {self.stitches_np.shape[0]}", 20, 20)
        dc.DrawText("Hold Left/Right/Up/Down arrows (Events throttled to ~60 FPS max)", 20, 40)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 15 (Input Throttling)", size=(1000, 700))
        self.canvas = ThrottledCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = [(0, 0, st[0]/10.0, st[1]/10.0, 50, 100, 200) for st in pattern.stitches]
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
