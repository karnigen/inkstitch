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

try:
    import pystitch as emb
    HAS_EMB = True
except ImportError:
    try:
        import pyembroidery as emb
        HAS_EMB = True
    except ImportError:
        HAS_EMB = False

class PesLoader:
    @staticmethod
    def load(path):
        if not HAS_EMB:
            raise RuntimeError("No embroidery parser library found!")

        pattern = emb.read(path)
        segments = []
        last_x, last_y = 0.0, 0.0
        cur_color = (200, 30, 30)

        for st in pattern.stitches:
            x, y = st[0] / 10.0, st[1] / 10.0
            cmd = st[2] if len(st) > 2 else 0

            # Jump stitch
            if hasattr(emb, 'JUMP') and cmd == emb.JUMP:
                last_x, last_y = x, y
                continue
            # End of pattern
            if hasattr(emb, 'END') and cmd == emb.END:
                break

            segments.append((last_x, last_y, x, y, cur_color[0], cur_color[1], cur_color[2]))
            last_x, last_y = x, y

        return segments

class MainFrame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None, title="PES Viewer - Step 3", size=(1000, 700))
        self.panel = wx.Panel(self)

        if initial_file and os.path.exists(initial_file):
            try:
                stitches = PesLoader.load(initial_file)
                print(f"Loaded {len(stitches)} stitches successfully.")
                self.SetTitle(f"PES Viewer - {os.path.basename(initial_file)} ({len(stitches)} stitches)")
            except Exception as e:
                print(f"Error loading file: {e}")

        self.Centre()
        self.Show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args = parser.parse_args()

    app = wx.App()
    MainFrame(initial_file=args.pes_file)
    app.MainLoop()