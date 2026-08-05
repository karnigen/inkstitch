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

class ColorCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.color_boundaries = [0]
        self.visible_count = 0

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.SetFocus()

    def jump_color_block(self, direction=1):
        if not self.color_boundaries:
            return

        curr = self.visible_count
        if direction > 0:
            target = next((b for b in self.color_boundaries if b > curr), self.stitches_np.shape[0])
        else:
            target = next((b for b in reversed(self.color_boundaries) if b < curr), 0)

        self.visible_count = target
        print(f"Jumped to color boundary stitch index: {self.visible_count}")
        self.Refresh()

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        ctrl_down = event.ControlDown()

        if ctrl_down and key == wx.WXK_RIGHT:
            self.jump_color_block(1)
        elif ctrl_down and key == wx.WXK_LEFT:
            self.jump_color_block(-1)
        else:
            event.Skip()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()
        dc.DrawText(f"Visible Stitches: {self.visible_count} / {self.stitches_np.shape[0]}", 20, 20)
        dc.DrawText(f"Color Boundaries: {self.color_boundaries}", 20, 40)
        dc.DrawText("Use Ctrl + Left / Right to jump between color blocks", 20, 60)

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 13 (Color Block Jumps)", size=(1000, 700))
        self.canvas = ColorCanvas(self)

        if initial_file and os.path.exists(initial_file):
            pattern = emb.read(initial_file)
            segs = []
            boundaries = [0]
            lx, ly = 0.0, 0.0

            for idx, st in enumerate(pattern.stitches):
                x, y = st[0]/10.0, st[1]/10.0
                cmd = st[2] if len(st) > 2 else 0

                # Check for color change command
                if hasattr(emb, 'COLOR_CHANGE') and cmd == emb.COLOR_CHANGE:
                    boundaries.append(len(segs))

                segs.append((lx, ly, x, y, 200, 100, 50))
                lx, ly = x, y

            self.canvas.stitches_np = np.array(segs, dtype=np.float32)
            self.canvas.color_boundaries = boundaries
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