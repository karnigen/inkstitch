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
import time
import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import pystitch as emb
    HAS_EMB = True
except ImportError:
    try:
        import pyembroidery as emb
        HAS_EMB = True
    except ImportError:
        HAS_EMB = False

if HAS_NUMBA:
    @numba.njit
    def render_grid_numba(buf, zoom, pan_x, pan_y):
        h, w, _ = buf.shape
        x_world_min = (-pan_x) / zoom
        x_world_max = (w - pan_x) / zoom
        y_world_min = (-pan_y) / zoom
        y_world_max = (h - pan_y) / zoom
        x_start = int(np.floor(x_world_min / 10.0) * 10)
        x_end = int(np.ceil(x_world_max / 10.0) * 10)
        y_start = int(np.floor(y_world_min / 10.0) * 10)
        y_end = int(np.ceil(y_world_max / 10.0) * 10)
        for xw in range(x_start, x_end+1, 10):
            sx = int(xw * zoom + pan_x)
            if sx < 0 or sx >= w: continue
            is_major = (xw % 50 == 0)
            is_axis = (xw == 0)
            if is_axis: r,g,b = 200, 100, 100
            elif is_major: r,g,b = 190, 190, 190
            else: r,g,b = 230, 230, 230
            for y in range(h):
                if is_axis:
                    buf[y, sx, 0] = r; buf[y, sx, 1] = g; buf[y, sx, 2] = b
                else:
                    if y % 3 != 0: continue
                    buf[y, sx, 0] = r; buf[y, sx, 1] = g; buf[y, sx, 2] = b
        for yw in range(y_start, y_end+1, 10):
            sy = int(yw * zoom + pan_y)
            if sy < 0 or sy >= h: continue
            is_major = (yw % 50 == 0)
            is_axis = (yw == 0)
            if is_axis: r,g,b = 100, 200, 100
            elif is_major: r,g,b = 190, 190, 190
            else: r,g,b = 230, 230, 230
            for x in range(w):
                if is_axis:
                    buf[sy, x, 0] = r; buf[sy, x, 1] = g; buf[sy, x, 2] = b
                else:
                    if x % 3 != 0: continue
                    buf[sy, x, 0] = r; buf[sy, x, 1] = g; buf[sy, x, 2] = b

    @numba.njit
    def render_shaded_numba(buf, stitches, visible_count, zoom, pan_x, pan_y, use_gradient, line_width):
        h, w, _ = buf.shape
        hw = line_width * 0.5
        lw_int = int(line_width)
        for i in range(visible_count):
            x1 = stitches[i,0] * zoom + pan_x
            y1 = stitches[i,1] * zoom + pan_y
            x2 = stitches[i,2] * zoom + pan_x
            y2 = stitches[i,3] * zoom + pan_y
            r_base = int(stitches[i,4]); g_base = int(stitches[i,5]); b_base = int(stitches[i,6])
            if (x1 < -200 and x2 < -200) or (x1 > w+200 and x2 > w+200): continue
            if (y1 < -200 and y2 < -200) or (y1 > h+200 and y2 > h+200): continue
            dx = x2 - x1; dy = y2 - y1
            length = int(np.sqrt(dx*dx + dy*dy)) + 1
            if length <= 0: continue
            r_dark = int(r_base * 0.35); g_dark = int(g_base * 0.35); b_dark = int(b_base * 0.35)
            r_light = int(r_base + (255 - r_base)*0.75); g_light = int(g_base + (255 - g_base)*0.75); b_light = int(b_base + (255 - b_base)*0.75)
            steps = length
            if steps < 1: steps = 1
            for s in range(steps+1):
                t = s / steps
                x = x1 + dx*t; y = y1 + dy*t
                if use_gradient:
                    r = int(r_dark + (r_light - r_dark)*t); g = int(g_dark + (g_light - g_dark)*t); b = int(b_dark + (b_light - b_dark)*t)
                else:
                    r = r_base; g = g_base; b = b_base
                if lw_int <= 1:
                    xi = int(x); yi = int(y)
                    if 0 <= xi < w and 0 <= yi < h:
                        buf[yi, xi, 0] = r; buf[yi, xi, 1] = g; buf[yi, xi, 2] = b
                else:
                    for oy in range(-lw_int, lw_int+1):
                        for ox in range(-lw_int, lw_int+1):
                            if ox*ox + oy*oy > hw*hw + 1: continue
                            xi = int(x + ox); yi = int(y + oy)
                            if 0 <= xi < w and 0 <= yi < h:
                                if ox*ox + oy*oy <= (hw-0.5)*(hw-0.5):
                                    buf[yi, xi, 0] = r; buf[yi, xi, 1] = g; buf[yi, xi, 2] = b
                                else:
                                    buf[yi, xi, 0] = (buf[yi, xi, 0] + r)//2
                                    buf[yi, xi, 1] = (buf[yi, xi, 1] + g)//2
                                    buf[yi, xi, 2] = (buf[yi, xi, 2] + b)//2


class ProgressBarPanel(wx.Panel):
    def __init__(self, parent, viewer_panel):
        super().__init__(parent, size=(-1, 58))
        self.viewer = viewer_panel
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(wx.Colour(250,250,250))
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnClick)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotionClick)
        self.dragging = False
        self.margin_x = 24
        self.bar_h = 14
        self.bar_y = 8
    def OnClick(self, e):
        self.dragging = True
        self.Seek(e.GetPosition().x)
        self.CaptureMouse()
    def OnLeftUp(self, e):
        if self.dragging:
            if self.HasCapture():
                self.ReleaseMouse()
            self.dragging = False
    def OnMotionClick(self, e):
        if self.dragging and e.Dragging() and e.LeftIsDown():
            self.Seek(e.GetPosition().x)
    def Seek(self, mouse_x):
        w = self.GetSize().width
        total = self.viewer.stitches_np.shape[0]
        if total == 0 or w == 0: return
        bar_w = w - 2*self.margin_x
        rel_x = mouse_x - self.margin_x
        ratio = max(0.0, min(1.0, rel_x / bar_w if bar_w>0 else 0))
        self.viewer.visible_count = int(ratio * total)
        self.viewer.need_redraw = True
        self.viewer.Refresh()
        self.Refresh()
    def OnPaint(self, e):
        dc = wx.PaintDC(self)
        w,h = self.GetSize()
        dc.SetBackground(wx.Brush(wx.Colour(250,250,250)))
        dc.Clear()
        total = self.viewer.stitches_np.shape[0]
        vis = self.viewer.visible_count
        bar_x = self.margin_x
        bar_y = self.bar_y
        bar_w = w - 2*self.margin_x
        bar_h = self.bar_h
        dc.SetBrush(wx.Brush(wx.Colour(230,230,230)))
        dc.SetPen(wx.Pen(wx.Colour(200,200,200), 1))
        dc.DrawRoundedRectangle(bar_x, bar_y, bar_w, bar_h, 4)
        if total == 0:
            dc.SetTextForeground(wx.Colour(120,120,120))
            dc.DrawText("No file loaded - open .pes", bar_x, bar_y+22)
            return
        stitches = self.viewer.stitches_np
        dc.SetClippingRegion(bar_x, bar_y, bar_w, bar_h)
        if total > 0:
            if bar_w < total:
                step = max(1, total // bar_w)
                for i in range(0, total, step):
                    r = int(stitches[i,4]); g = int(stitches[i,5]); b = int(stitches[i,6])
                    xi = bar_x + int((i/total)*bar_w)
                    dc.SetPen(wx.Pen(wx.Colour(r,g,b), 1))
                    dc.DrawLine(xi, bar_y, xi, bar_y+bar_h)
            else:
                last_color = None
                block_start = 0
                for i in range(total):
                    r = int(stitches[i,4]); g = int(stitches[i,5]); b = int(stitches[i,6])
                    col = (r,g,b)
                    if col != last_color and last_color is not None:
                        x0 = bar_x + int((block_start/total)*bar_w)
                        x1 = bar_x + int((i/total)*bar_w)
                        dc.SetBrush(wx.Brush(wx.Colour(last_color[0], last_color[1], last_color[2])))
                        dc.SetPen(wx.Pen(wx.Colour(last_color[0], last_color[1], last_color[2]), 1))
                        dc.DrawRectangle(x0, bar_y, max(2, x1-x0), bar_h)
                        block_start = i
                    last_color = col
                if last_color:
                    x0 = bar_x + int((block_start/total)*bar_w)
                    dc.SetBrush(wx.Brush(wx.Colour(last_color[0], last_color[1], last_color[2])))
                    dc.SetPen(wx.Pen(wx.Colour(last_color[0], last_color[1], last_color[2]), 1))
                    dc.DrawRectangle(x0, bar_y, bar_w - (x0-bar_x), bar_h)
        progress_w = int((vis/total)*bar_w) if total>0 else 0
        dc.SetBrush(wx.Brush(wx.Colour(255,255,255,150)))
        dc.SetPen(wx.TRANSPARENT_PEN)
        if progress_w < bar_w:
            dc.DrawRectangle(bar_x+progress_w, bar_y, bar_w-progress_w, bar_h)
        dc.DestroyClippingRegion()
        knob_x = bar_x + progress_w
        dc.SetPen(wx.Pen(wx.Colour(40,40,40), 2))
        dc.SetBrush(wx.Brush(wx.Colour(250,250,250)))
        dc.DrawCircle(knob_x, bar_y + bar_h//2, 6)
        dc.SetBrush(wx.Brush(wx.Colour(40,40,40)))
        dc.DrawCircle(knob_x, bar_y + bar_h//2, 3)
        dc.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        dc.SetTextForeground(wx.Colour(30,30,30))
        txt_left = f"{vis}/{total} stitches"
        txt_center = f"{vis/total*100:.1f}%" if total>0 else "0%"
        if hasattr(self.viewer, 'bounds') and self.viewer.bounds != (0,0,0,0):
            bw = self.viewer.bounds[2]-self.viewer.bounds[0]
            bh = self.viewer.bounds[3]-self.viewer.bounds[1]
            txt_right = f"{bw:.1f} x {bh:.1f} mm | {len(self.viewer.color_boundaries)} colors"
        else:
            txt_right = ""
        dc.DrawText(txt_left, bar_x, bar_y+bar_h+6)
        tw,_ = dc.GetTextExtent(txt_center)
        dc.DrawText(txt_center, bar_x + (bar_w-tw)//2, bar_y+bar_h+6)
        if txt_right:
            tw,_ = dc.GetTextExtent(txt_right)
            dc.DrawText(txt_right, bar_x + bar_w - tw, bar_y+bar_h+6)


class PesViewerFastPanel(wx.Panel):
    def __init__(self, parent, progress_bar):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.zoom = 1.0
        self.pan_x, self.pan_y = 400, 300
        self.drag_start = None
        self.pan_start = (0,0)
        self.line_width = 2.0
        self.visible_count = 0
        self.step_size = 10
        self.show_grid = True
        self.stitches_np = np.zeros((0,7), dtype=np.float32)
        self.bounds = (0,0,0,0)
        self.color_boundaries = []
        self.cached_bitmap = None
        self.need_redraw = True
        self.progress_bar = progress_bar
        self._last_key_time = 0
        self._key_throttle = 0.03
        self._last_dir = 1
        self.play_timer = wx.Timer(self)
        self.play_speed = 20
        self.play_step = 10
        self.is_playing = False
        self.Bind(wx.EVT_TIMER, self.OnPlayTimer, self.play_timer)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_KEY_UP, self.OnKeyUp)
        self.SetFocus()
    def OnSize(self, e):
        self.need_redraw = True
        e.Skip()
    def FitToScreen(self):
        if self.stitches_np.shape[0] == 0: return
        min_x, min_y, max_x, max_y = self.bounds
        bw = max_x - min_x; bh = max_y - min_y
        if bw < 1: bw = 1
        if bh < 1: bh = 1
        w,h = self.GetSize()
        if w < 10 or h < 10: w,h = 1200, 800
        zoom_x = (w * 0.8) / bw; zoom_y = (h * 0.8) / bh
        self.zoom = min(zoom_x, zoom_y)
        cx = (min_x + max_x)/2; cy = (min_y + max_y)/2
        self.pan_x = w/2 - cx * self.zoom
        self.pan_y = h/2 - cy * self.zoom
        self.need_redraw = True
        self.Refresh()
        if self.progress_bar: self.progress_bar.Refresh()
    def OnPlayTimer(self, e):
        total = self.stitches_np.shape[0]
        if total == 0:
            self.play_timer.Stop(); self.is_playing=False; return
        self.visible_count += self.play_step * self._last_dir
        if self.visible_count >= total:
            self.visible_count = total; self.play_timer.Stop(); self.is_playing=False
        elif self.visible_count <= 0:
            self.visible_count = 0; self.play_timer.Stop(); self.is_playing=False
        self.need_redraw = True; self.Refresh()
        if self.progress_bar: self.progress_bar.Refresh()
    def ToggleAutoPlay(self, forward=True):
        if self.is_playing:
            self.play_timer.Stop(); self.is_playing=False
        else:
            self._last_dir = 1 if forward else -1
            self.play_timer.Start(self.play_speed); self.is_playing=True
    def OnKeyUp(self, e):
        self._last_key_time = 0
        e.Skip()
    def JumpToColor(self, direction):
        if not self.color_boundaries: return
        cur = self.visible_count
        if direction > 0:
            for b in self.color_boundaries:
                if b > cur:
                    self.visible_count = b
                    return
            self.visible_count = self.stitches_np.shape[0]
        else:
            prev = 0
            for b in self.color_boundaries:
                if b < cur:
                    prev = b
                else:
                    break
            if cur in self.color_boundaries:
                idx = self.color_boundaries.index(cur)
                if idx > 0:
                    self.visible_count = self.color_boundaries[idx-1]
                else:
                    self.visible_count = 0
            else:
                self.visible_count = prev
    def OnKeyDown(self, e):
        now = time.time()
        key = e.GetKeyCode()
        is_space_or_c = key in (wx.WXK_SPACE, 67, 99, 86, 118)
        if not is_space_or_c and now - self._last_key_time < self._key_throttle:
            if not e.AltDown() and not e.ControlDown():
                return
        self._last_key_time = now
        total = self.stitches_np.shape[0]
        is_alt = e.AltDown()
        is_ctrl = e.ControlDown()
        changed = False
        step = 1 if is_alt else self.step_size
        if is_ctrl and key in (wx.WXK_RIGHT, wx.WXK_LEFT):
            if key == wx.WXK_RIGHT:
                self.JumpToColor(1); self._last_dir = 1
            else:
                self.JumpToColor(-1); self._last_dir = -1
            changed = True
        elif key == wx.WXK_RIGHT:
            if self.visible_count < total:
                self.visible_count = min(total, self.visible_count + step)
                self._last_dir = 1; changed = True
        elif key == wx.WXK_LEFT:
            if self.visible_count > 0:
                self.visible_count = max(0, self.visible_count - step)
                self._last_dir = -1; changed = True
        elif key == wx.WXK_UP:
            self.visible_count = min(total, self.visible_count + step * 10)
            self._last_dir = 1; changed = True
        elif key == wx.WXK_DOWN:
            self.visible_count = max(0, self.visible_count - step * 10)
            self._last_dir = -1; changed = True
        elif key == wx.WXK_HOME:
            self.visible_count = 0; changed = True
        elif key == wx.WXK_END:
            self.visible_count = total; changed = True
        elif key == wx.WXK_SPACE:
            self.ToggleAutoPlay(forward=self._last_dir>0)
            return
        elif key in (43, 61, 388, wx.WXK_NUMPAD_ADD):
            self.line_width = min(10.0, self.line_width + 0.5); changed = True
        elif key in (45, 95, 390, wx.WXK_NUMPAD_SUBTRACT):
            self.line_width = max(0.5, self.line_width - 0.5); changed = True
        elif key == 67 or key == 99:
            self.ToggleAutoPlay(forward=True)
            return
        elif key == 86 or key == 118:
            self.ToggleAutoPlay(forward=False)
            return
        elif key == 70 or key == 102:
            self.FitToScreen(); return
        elif key == 71 or key == 103:
            self.show_grid = not self.show_grid; changed = True
        elif key == 72 or key == 104:
            self.ShowHelp(); return
        elif key == wx.WXK_ESCAPE:
            if self.is_playing:
                self.play_timer.Stop(); self.is_playing=False
                return
        if changed:
            if self.is_playing and key in (wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_UP, wx.WXK_DOWN, wx.WXK_HOME, wx.WXK_END) and not is_ctrl:
                self.play_timer.Stop(); self.is_playing=False
            self.need_redraw = True; self.Refresh()
            if self.progress_bar: self.progress_bar.Refresh()
        else:
            e.Skip()
    def ShowHelp(self):
        help_text = "PES Viewer PRO\n\nMouse: Wheel=Zoom Drag=Pan Click bar=Seek\n\nPlayback:\n  Right/Left - +/- step\n  Alt+Right/Left - +/- 1\n  Ctrl+Right/Left - Next/Prev color\n  Up/Down - Fast\n  Home/End - First/Last\n  Space - Play/Pause toggle\n  C - Forward  V - Backward\n  Esc - Stop\n\nView: +/- width F=fit G=grid H=help\n"
        dlg = wx.MessageDialog(self, help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal(); dlg.Destroy()
    def SetStepSize(self, size): self.step_size = max(1, size)
    def LoadPes(self, path, fit_to_screen=True):
        if not HAS_EMB:
            wx.MessageBox("pystitch not installed", "Error"); return False
        try:
            pattern = emb.read(path)
        except Exception as ex:
            wx.MessageBox(f"Failed to load PES: {ex}", "Error"); return False
        segs = []; last_x=last_y=0; cur_color_idx=0
        palette = pattern.threadlist if hasattr(pattern,'threadlist') and pattern.threadlist else [(220,30,30)]
        min_x=min_y=1e9; max_x=max_y=-1e9
        self.color_boundaries = [0]
        for st in pattern.stitches:
            x = st[0]/10.0; y = st[1]/10.0; cmd = st[2] if len(st)>2 else 0
            if hasattr(emb,'JUMP') and cmd == emb.JUMP:
                last_x,last_y=x,y; continue
            if hasattr(emb,'COLOR_CHANGE') and (cmd == emb.COLOR_CHANGE or (cmd & 0x04)):
                cur_color_idx+=1
                if segs:
                    self.color_boundaries.append(len(segs))
                last_x,last_y=x,y; continue
            if hasattr(emb,'END') and cmd == emb.END: break
            if cur_color_idx < len(palette):
                col = palette[cur_color_idx]
                if hasattr(col,'get_red'): rgb=(col.get_red(), col.get_green(), col.get_blue())
                elif isinstance(col,(list,tuple)): rgb=tuple(col[:3])
                else: rgb=(220,30,30)
            else: rgb=(220,30,30)
            segs.append((last_x,last_y,x,y,rgb[0],rgb[1],rgb[2]))
            min_x=min(min_x,last_x,x); min_y=min(min_y,last_y,y)
            max_x=max(max_x,last_x,x); max_y=max(max_y,last_y,y)
            last_x,last_y=x,y
        if segs:
            self.stitches_np = np.array(segs, dtype=np.float32)
            self.bounds=(min_x,min_y,max_x,max_y)
            self.visible_count=self.stitches_np.shape[0]
            self.color_boundaries = sorted(set(self.color_boundaries))
            if fit_to_screen:
                wx.CallAfter(self.FitToScreen)
        self.need_redraw=True; self.Refresh()
        if self.progress_bar: self.progress_bar.Refresh()
        self.SetFocus()
        return True
    def OnPaint(self, e):
        dc = wx.PaintDC(self); dc.Clear()
        if not self.need_redraw and self.cached_bitmap:
            dc.DrawBitmap(self.cached_bitmap,0,0); return
        w,h = self.GetSize()
        if self.stitches_np.shape[0]==0:
            dc.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            dc.DrawText("Open .PES via File > Open or pass as cmd arg",20,20)
            dc.DrawText("H=help, Space=play/pause, Ctrl+Arrows=color, Alt+Arrows=1",20,45)
            return
        use_gradient = self.zoom > 1.2
        buf = np.full((h,w,3), 255, dtype=np.uint8)
        if HAS_NUMBA:
            if self.show_grid:
                render_grid_numba(buf, self.zoom, self.pan_x, self.pan_y)
            if self.stitches_np.shape[0]>0 and self.visible_count>0:
                render_shaded_numba(buf, self.stitches_np, self.visible_count, self.zoom, self.pan_x, self.pan_y, use_gradient, self.line_width)
        img = wx.Image(w,h); img.SetData(buf.tobytes()); bmp = wx.Bitmap(img)
        self.cached_bitmap=bmp; self.need_redraw=False; dc.DrawBitmap(bmp,0,0)
    def OnWheel(self,e):
        mx,my=e.GetPosition(); old=self.zoom
        self.zoom*=1.15 if e.GetWheelRotation()>0 else 1/1.15
        self.zoom=max(0.05,min(50.0,self.zoom))
        scale=self.zoom/old
        self.pan_x=mx-scale*(mx-self.pan_x); self.pan_y=my-scale*(my-self.pan_y)
        self.need_redraw=True; self.Refresh()
    def OnLeftDown(self,e):
        self.drag_start=e.GetPosition(); self.pan_start=(self.pan_x,self.pan_y); self.CaptureMouse(); self.SetFocus()
    def OnLeftUp(self,e):
        if self.HasCapture(): self.ReleaseMouse()
        self.drag_start=None
        if self.progress_bar and self.progress_bar.dragging:
            self.progress_bar.dragging=False
            if self.progress_bar.HasCapture(): self.progress_bar.ReleaseMouse()
    def OnMotion(self,e):
        if self.drag_start and e.Dragging() and e.LeftIsDown():
            dx=e.GetPosition()[0]-self.drag_start[0]; dy=e.GetPosition()[1]-self.drag_start[1]
            self.pan_x,self.pan_y=self.pan_start[0]+dx,self.pan_start[1]+dy
            self.need_redraw=True; self.Refresh()

class Frame(wx.Frame):
    def __init__(self, initial_file=None):
        super().__init__(None,title="PES Viewer PRO v2.2",size=(1200,980))
        main_panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.viewer = PesViewerFastPanel(main_panel, None)
        self.progress = ProgressBarPanel(main_panel, self.viewer)
        self.viewer.progress_bar = self.progress
        self.progress.Bind(wx.EVT_LEFT_UP, self.viewer.OnLeftUp)
        sizer.Add(self.viewer, 1, wx.EXPAND)
        sizer.Add(self.progress, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        main_panel.SetSizer(sizer)
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        openItem = fileMenu.Append(wx.ID_OPEN, "Open .PES\tCtrl+O")
        fitItem = fileMenu.Append(wx.ID_ANY, "Fit to screen\tF")
        gridItem = fileMenu.AppendCheckItem(wx.ID_ANY, "Show 1cm grid\tG"); gridItem.Check(True)
        helpItem = fileMenu.Append(wx.ID_ANY, "Help\tH")
        menubar.Append(fileMenu, "File")
        playbackMenu = wx.Menu()
        s1 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 1 (Alt+Arrows)")
        s10 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 10"); s10.Check(True)
        s50 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 50")
        s100 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 100")
        s500 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 500")
        playbackMenu.AppendSeparator()
        playItem = playbackMenu.Append(wx.ID_ANY, "Play/Pause\tSpace")
        nextCol = playbackMenu.Append(wx.ID_ANY, "Next color\tCtrl+Right")
        prevCol = playbackMenu.Append(wx.ID_ANY, "Prev color\tCtrl+Left")
        menubar.Append(playbackMenu, "Playback")
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.OnOpen, openItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.FitToScreen(), fitItem)
        self.Bind(wx.EVT_MENU, self.OnToggleGrid, gridItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.ShowHelp(), helpItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(1), s1)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(10), s10)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(50), s50)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(100), s100)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(500), s500)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.ToggleAutoPlay(True), playItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.JumpToColor(1) or self._refresh_after_color_jump(), nextCol)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.JumpToColor(-1) or self._refresh_after_color_jump(), prevCol)
        self.CreateStatusBar()
        self.SetStatusText("Space=play/pause | Ctrl+Arrows=color | Alt+Arrows=1 | F=fit G=grid H=help")
        self.Centre(); self.Show()
        if initial_file and os.path.exists(initial_file):
            if self.viewer.LoadPes(initial_file, fit_to_screen=True):
                total=self.viewer.stitches_np.shape[0]
                bw=self.viewer.bounds[2]-self.viewer.bounds[0]; bh=self.viewer.bounds[3]-self.viewer.bounds[1]
                self.SetTitle(f"PES Viewer PRO v2.2 - {os.path.basename(initial_file)} - {total} sts")
    def _refresh_after_color_jump(self):
        self.viewer.need_redraw=True; self.viewer.Refresh(); self.progress.Refresh()
    def OnToggleGrid(self, e):
        self.viewer.show_grid=e.IsChecked(); self.viewer.need_redraw=True; self.viewer.Refresh()
    def OnOpen(self,e):
        dlg=wx.FileDialog(self,"Open PES",wildcard="PES (*.pes)|*.pes|All|*.*",style=wx.FD_OPEN)
        if dlg.ShowModal()==wx.ID_OK:
            path=dlg.GetPath()
            if self.viewer.LoadPes(path, fit_to_screen=True):
                total=self.viewer.stitches_np.shape[0]
                bw=self.viewer.bounds[2]-self.viewer.bounds[0]; bh=self.viewer.bounds[3]-self.viewer.bounds[1]
                self.SetTitle(f"PES Viewer PRO v2.2 - {os.path.basename(path)} - {total} sts - {bw:.1f}x{bh:.1f}mm")
                self.progress.Refresh()
        dlg.Destroy()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="PES Viewer PRO v2.2")
    parser.add_argument("pes_file", nargs="?", help="Input .pes file")
    args=parser.parse_args()
    app=wx.App()
    Frame(initial_file=args.pes_file)
    app.MainLoop()
