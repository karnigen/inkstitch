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

class AnimatedCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.visible_count = 0

        # Playback Timer Setup
        self.timer = wx.Timer(self)
        self.play_speed = 5  # Stitches per tick
        self.play_direction = 1

        self.Bind(wx.EVT_TIMER, self.OnTimerTick, self.timer)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.SetFocus()

    def TogglePlay(self, direction=1, speed=5):
        if self.timer.IsRunning() and self.play_direction == direction:
            self.timer.Stop()
            print("[PLAYBACK] Paused")
        else:
            self.play_direction = direction
            self.play_speed = speed
            self.timer.Start(20)  # ~50 FPS timer interval
            print(f"[PLAYBACK] Started (Dir: {direction}, Speed: {speed})")

    def OnTimerTick(self, event):
        total = self.stitches_np.shape[0]
        new_count = self.visible_count + (self.play_direction * self.play_speed)

        if new_count >= total:
            self.visible_count = total
            self.timer.Stop()
        elif new_count <= 0:
            self.visible_count = 0
            self.timer.Stop()
        else:
            self.visible_count = new_count

        self.Refresh()

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        if key == wx.WXK_SPACE:
            self.TogglePlay(direction=1, speed=5)
        elif key in (ord('V'), ord('v')):
            self.TogglePlay(direction=1, speed=25)  # Fast Forward
        elif key in (ord('C'), ord('c')):
            self.TogglePlay(direction=-1, speed=10) # Rewind
        elif key == wx.WXK_ESCAPE:
            if self.timer.IsRunning():
                self.timer.Stop()
        else:
            event.Skip()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()
        status = "PLAYING" if self.timer.IsRunning() else "PAUSED"
        dc.DrawText(f"Status: {status} | Visible: {self.visible_count} / {self.stitches_np.shape[0]}", 20, 20)
        dc.DrawText("Space: Play/Pause | V: Fast Forward | C: Rewind | Esc: Stop", 20, 40)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 14 (Auto-Play Simulator)", size=(1000, 700))
        self.canvas = AnimatedCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = [(0, 0, st[0]/10.0, st[1]/10.0, 100, 200, 50) for st in pattern.stitches]
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
