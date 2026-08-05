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

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="PES Viewer - Step 19 (MenuBar Integration)", size=(1000, 700))
        self.SetupMenuBar()

        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(wx.Colour(240, 240, 240))

        self.CreateStatusBar()
        self.SetStatusText("Ready. Load a .pes file via File -> Open")

        self.Centre()
        self.Show()

    def SetupMenuBar(self):
        menubar = wx.MenuBar()

        # File Menu
        file_menu = wx.Menu()
        open_item = file_menu.Append(wx.ID_OPEN, "&Open...\tCtrl+O", "Open PES File")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "E&xit\tCtrl+Q", "Exit Application")

        # View Menu
        view_menu = wx.Menu()
        fit_item = view_menu.Append(wx.ID_ANY, "&Fit to Screen\tF", "Fit pattern to window")
        grid_item = view_menu.Append(wx.ID_ANY, "Toggle &Grid\tG", "Toggle millimeter grid")

        # Playback Menu
        play_menu = wx.Menu()
        play_item = play_menu.Append(wx.ID_ANY, "&Play / Pause\tSpace", "Toggle playback")

        menubar.Append(file_menu, "&File")
        menubar.Append(view_menu, "&View")
        menubar.Append(play_menu, "&Playback")

        self.SetMenuBar(menubar)

        self.Bind(wx.EVT_MENU, self.OnOpen, open_item)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), exit_item)

    def OnOpen(self, event):
        with wx.FileDialog(self, "Open PES File", wildcard="PES files (*.pes)|*.pes",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.SetStatusText(f"Loaded: {path}")

if __name__ == "__main__":
    app = wx.App()
    MainFrame()
    app.MainLoop()
