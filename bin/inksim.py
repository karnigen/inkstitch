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

# Safe Numba import pattern
try:
    from numba import njit
    HAS_NUMBA = True
    print("[INFO] Numba is available. JIT acceleration enabled.")
except ImportError:
    HAS_NUMBA = False
    print("[WARN] Numba not found. Running in pure-Python/NumPy fallback mode.")
    # Dummy decorator fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class NumbaFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title=f"PES Viewer - Step 9 (Numba Integration: {HAS_NUMBA})", size=(1000, 700))

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        status_txt = f"Numba Status: {'ENABLED (JIT Active)' if HAS_NUMBA else 'DISABLED (Fallback Mode)'}"
        label = wx.StaticText(panel, label=status_txt)
        font = label.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetFont(font)

        sizer.Add(label, 0, wx.ALL | wx.ALIGN_CENTER, 20)
        panel.SetSizer(sizer)

        self.Centre()
        self.Show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args = parser.parse_args()

    app = wx.App()
    NumbaFrame(initial_file=args.pes_file)
    app.MainLoop()