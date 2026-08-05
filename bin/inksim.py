#!/usr/bin/env python3

from pathlib import Path
import sys
import os
import argparse

#-------------------------------------------------------------------
# Dual run: system python3 or .venv python

script_dir = Path(__file__).resolve().parent

# !!! Hardcoded relative path to the virtual environment directory - adjust if you move this script.
venv_dir = (script_dir / ".." / ".venv").resolve()
APP_TITLE = "InkSim"

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

# restart the virtual environment if not already active
ensure_venv()
#-------------------------------------------------------------------

import wx
import time
import numpy as np
import numba
import pystitch as emb

@numba.njit
def render_grid_numba(buf, zoom, pan_x, pan_y):
    # Draw 1 cm helper grid into the RGB buffer.
    # - every 10 mm: minor grid line
    # - every 50 mm: major grid line
    # - x=0 / y=0: highlighted axes
    # zoom converts mm -> pixels, pan is screen-space origin offset.
    h, w, _ = buf.shape

    # World-space area currently visible in the viewport.
    x_world_min = (-pan_x) / zoom
    x_world_max = (w - pan_x) / zoom
    y_world_min = (-pan_y) / zoom
    y_world_max = (h - pan_y) / zoom

    # Snap bounds to full 10 mm steps so edge lines are still drawn.
    x_start = int(np.floor(x_world_min / 10.0) * 10)
    x_end = int(np.ceil(x_world_max / 10.0) * 10)
    y_start = int(np.floor(y_world_min / 10.0) * 10)
    y_end = int(np.ceil(y_world_max / 10.0) * 10)

    # Vertical lines.
    for xw in range(x_start, x_end+1, 10):
        # Project world x to screen x.
        sx = int(xw * zoom + pan_x)
        if sx < 0 or sx >= w: continue

        # Choose line style.
        is_major = (xw % 50 == 0)
        is_axis = (xw == 0)

        if is_axis: r,g,b = 200, 100, 100      # red axis
        elif is_major: r,g,b = 190, 190, 190   # major line
        else: r,g,b = 230, 230, 230            # minor line

        # Keep minor/major lines subtle using a dotted pattern.
        for y in range(h):
            if is_axis:
                buf[y, sx, 0] = r
                buf[y, sx, 1] = g
                buf[y, sx, 2] = b
            else:
                if y % 3 != 0: continue
                buf[y, sx, 0] = r
                buf[y, sx, 1] = g
                buf[y, sx, 2] = b

    # Horizontal lines (same logic as vertical).
    for yw in range(y_start, y_end+1, 10):
        sy = int(yw * zoom + pan_y)
        if sy < 0 or sy >= h: continue
        is_major = (yw % 50 == 0)
        is_axis = (yw == 0)

        if is_axis: r,g,b = 100, 200, 100      # green axis
        elif is_major: r,g,b = 190, 190, 190   # major line
        else: r,g,b = 230, 230, 230            # minor line

        for x in range(w):
            if is_axis:
                buf[sy, x, 0] = r
                buf[sy, x, 1] = g
                buf[sy, x, 2] = b
            else:
                if x % 3 != 0: continue
                buf[sy, x, 0] = r
                buf[sy, x, 1] = g
                buf[sy, x, 2] = b

@numba.njit
def render_shaded_numba(
    buf,
    stitches,
    visible_count,
    zoom,
    pan_x,
    pan_y,
    use_gradient,
    line_width,
    dark_factor,
    light_factor,
):
    # Draw visible stitch segments into the RGB buffer.
    # Each segment is [x1, y1, x2, y2, r, g, b] in mm + base thread color.
    # We project mm -> screen pixels using zoom/pan and then rasterize.
    h, w, _ = buf.shape
    # The configured width is measured at zoom 1.0; scale it with the
    # world-to-screen transform so thread thickness follows the design.
    effective_line_width = line_width * zoom
    hw = effective_line_width * 0.5
    lw_int = int(effective_line_width)

    for i in range(visible_count):
        # Convert segment endpoints from world space (mm) to screen pixels.
        x1 = stitches[i,0] * zoom + pan_x
        y1 = stitches[i,1] * zoom + pan_y
        x2 = stitches[i,2] * zoom + pan_x
        y2 = stitches[i,3] * zoom + pan_y

        # Get base thread color for this segment.
        r_base = int(stitches[i, 4])
        g_base = int(stitches[i, 5])
        b_base = int(stitches[i, 6])

        # Cheap reject: ignore segments completely far outside the viewport.
        if (x1 < -200 and x2 < -200) or (x1 > w+200 and x2 > w+200): continue
        if (y1 < -200 and y2 < -200) or (y1 > h+200 and y2 > h+200): continue

        # Compute the segment length in pixels and sample points along it.
        dx = x2 - x1
        dy = y2 - y1
        length = int(np.sqrt(dx*dx + dy*dy)) + 1
        if length <= 0: continue

        # Precompute dark/light variants for pseudo-3D thread shading.
        r_dark = int(r_base * dark_factor)
        g_dark = int(g_base * dark_factor)
        b_dark = int(b_base * dark_factor)
        r_light = int(r_base + (255 - r_base) * light_factor)
        g_light = int(g_base + (255 - g_base) * light_factor)
        b_light = int(b_base + (255 - b_base) * light_factor)

        steps = length
        steps = max(steps, 1)
        for s in range(steps+1):
            t = s / steps
            x = x1 + dx * t
            y = y1 + dy * t

            # Optional gradient along the segment to make stitches look less flat.
            if use_gradient:
                r = int(r_dark + (r_light - r_dark) * t)
                g = int(g_dark + (g_light - g_dark) * t)
                b = int(b_dark + (b_light - b_dark) * t)
            else:
                r = r_base
                g = g_base
                b = b_base

            # Fast path for thin lines (single pixel footprint).
            if lw_int <= 1:
                xi = int(x)
                yi = int(y)
                if 0 <= xi < w and 0 <= yi < h:
                    buf[yi, xi, 0] = r
                    buf[yi, xi, 1] = g
                    buf[yi, xi, 2] = b
            else:
                # Thick lines: draw a disk around each sampled point.
                # Interior pixels are solid, rim pixels are blended for a softer edge.
                for oy in range(-lw_int, lw_int+1):
                    for ox in range(-lw_int, lw_int+1):
                        if ox*ox + oy*oy > hw*hw + 1: continue
                        xi = int(x + ox)
                        yi = int(y + oy)
                        if 0 <= xi < w and 0 <= yi < h:
                            if ox*ox + oy*oy <= (hw-0.5)*(hw-0.5):
                                buf[yi, xi, 0] = r
                                buf[yi, xi, 1] = g
                                buf[yi, xi, 2] = b
                            else:
                                buf[yi, xi, 0] = (buf[yi, xi, 0] + r)//2
                                buf[yi, xi, 1] = (buf[yi, xi, 1] + g)//2
                                buf[yi, xi, 2] = (buf[yi, xi, 2] + b)//2


def get_supported_input_wildcard():
    """Build a wx file filter from the formats readable by pystitch."""
    extensions = set()
    for file_type in emb.EmbPattern.supported_formats():
        if file_type.get("reader") is None:
            continue
        file_extensions = file_type.get("extensions", ())
        if isinstance(file_extensions, str):
            file_extensions = (file_extensions,)
        extensions.update(ext.lstrip(".").lower() for ext in file_extensions)

    patterns = ";".join(f"*.{ext}" for ext in sorted(extensions))
    return f"Embroidery files ({patterns})|{patterns}|All files|*.*"

class ProgressBarPanel(wx.Panel):
    """Interactive stitch timeline shown below the embroidery viewer.

    The panel uses the viewer as its source of truth. It displays the loaded
    stitch colors, overlays the currently visible portion, and lets the user
    seek by clicking or dragging across the timeline.
    """

    def __init__(self, parent, viewer_panel):
        """Create a timeline connected to ``viewer_panel``."""
        super().__init__(parent, size=(-1, 58))
        self.viewer = viewer_panel
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(wx.Colour(250, 250, 250))
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnClick)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotionClick)

        self.dragging = False
        self.margin_x = 24
        self.bar_y = 8
        self.bar_h = 14

    def OnClick(self, e):
        """Start seeking at the mouse position."""
        self.dragging = True
        self.Seek(e.GetPosition().x)
        self.CaptureMouse()

    def OnLeftUp(self, e):
        """Finish a seek operation and release the mouse capture."""
        if self.dragging:
            if self.HasCapture():
                self.ReleaseMouse()
            self.dragging = False

    def OnMotionClick(self, e):
        """Update the seek position while the left button is dragged."""
        if self.dragging and e.Dragging() and e.LeftIsDown():
            self.Seek(e.GetPosition().x)

    def Seek(self, mouse_x):
        """Map a horizontal mouse position to a visible stitch count."""
        w = self.GetSize().width
        total = self.viewer.stitches_np.shape[0]
        if total == 0 or w == 0:
            return

        bar_w = w - 2*self.margin_x
        rel_x = mouse_x - self.margin_x
        ratio = max(0.0, min(1.0, rel_x / bar_w if bar_w > 0 else 0))
        self.viewer.visible_count = int(ratio * total)
        self.viewer.need_redraw = True
        self.viewer.Refresh()
        self.Refresh()

    def OnPaint(self, e):
        """Paint the color timeline, progress overlay, knob, and labels."""
        dc = wx.PaintDC(self)
        w, _ = self.GetSize()
        dc.SetBackground(wx.Brush(wx.Colour(250, 250, 250)))
        dc.Clear()

        total = self.viewer.stitches_np.shape[0]
        vis = self.viewer.visible_count
        bar_x = self.margin_x
        bar_y = self.bar_y
        bar_w = w - 2*self.margin_x
        bar_h = self.bar_h
        dc.SetBrush(wx.Brush(wx.Colour(230, 230, 230)))
        dc.SetPen(wx.Pen(wx.Colour(200, 200, 200), 1))
        dc.DrawRoundedRectangle(bar_x, bar_y, bar_w, bar_h, 4)

        if total == 0:
            dc.SetTextForeground(wx.Colour(120, 120, 120))
            dc.DrawText("No file loaded - open an embroidery file", bar_x, bar_y + 22)
            return

        stitches = self.viewer.stitches_np
        dc.SetClippingRegion(bar_x, bar_y, bar_w, bar_h)
        if bar_w < total:
            step = max(1, total // bar_w)
            for i in range(0, total, step):
                r = int(stitches[i, 4])
                g = int(stitches[i, 5])
                b = int(stitches[i, 6])
                xi = bar_x + int((i / total) * bar_w)
                dc.SetPen(wx.Pen(wx.Colour(r, g, b), 1))
                dc.DrawLine(xi, bar_y, xi, bar_y + bar_h)
        else:
            last_color = None
            block_start = 0
            for i in range(total):
                r = int(stitches[i, 4])
                g = int(stitches[i, 5])
                b = int(stitches[i, 6])
                color = (r, g, b)
                if color != last_color and last_color is not None:
                    x0 = bar_x + int((block_start / total) * bar_w)
                    x1 = bar_x + int((i / total) * bar_w)
                    wx_color = wx.Colour(*last_color)
                    dc.SetBrush(wx.Brush(wx_color))
                    dc.SetPen(wx.Pen(wx_color, 1))
                    dc.DrawRectangle(x0, bar_y, max(2, x1 - x0), bar_h)
                    block_start = i
                last_color = color

            if last_color:
                x0 = bar_x + int((block_start / total) * bar_w)
                wx_color = wx.Colour(*last_color)
                dc.SetBrush(wx.Brush(wx_color))
                dc.SetPen(wx.Pen(wx_color, 1))
                dc.DrawRectangle(x0, bar_y, bar_w - (x0 - bar_x), bar_h)

        progress_w = int((vis / total) * bar_w)
        dc.SetBrush(wx.Brush(wx.Colour(255, 255, 255, 150)))
        dc.SetPen(wx.TRANSPARENT_PEN)
        if progress_w < bar_w:
            dc.DrawRectangle(bar_x + progress_w, bar_y, bar_w - progress_w, bar_h)
        dc.DestroyClippingRegion()

        knob_x = bar_x + progress_w
        dc.SetPen(wx.Pen(wx.Colour(40, 40, 40), 2))
        dc.SetBrush(wx.Brush(wx.Colour(250, 250, 250)))
        dc.DrawCircle(knob_x, bar_y + bar_h // 2, 6)
        dc.SetBrush(wx.Brush(wx.Colour(40, 40, 40)))
        dc.DrawCircle(knob_x, bar_y + bar_h // 2, 3)
        dc.SetFont(wx.Font(
            9,
            wx.FONTFAMILY_DEFAULT,
            wx.FONTSTYLE_NORMAL,
            wx.FONTWEIGHT_NORMAL,
        ))
        dc.SetTextForeground(wx.Colour(30, 30, 30))

        txt_left = f"{vis}/{total} stitches"
        txt_center = f"{vis / total * 100:.1f}%"
        if hasattr(self.viewer, "bounds") and self.viewer.bounds != (0, 0, 0, 0):
            bw = self.viewer.bounds[2] - self.viewer.bounds[0]
            bh = self.viewer.bounds[3] - self.viewer.bounds[1]
            txt_right = (
                f"{bw:.1f} x {bh:.1f} mm | "
                f"{len(self.viewer.color_boundaries)} colors"
            )
        else:
            txt_right = ""

        text_y = bar_y + bar_h + 6
        dc.DrawText(txt_left, bar_x, text_y)
        tw, _ = dc.GetTextExtent(txt_center)
        dc.DrawText(txt_center, bar_x + (bar_w - tw) // 2, text_y)
        if txt_right:
            tw, _ = dc.GetTextExtent(txt_right)
            dc.DrawText(txt_right, bar_x + bar_w - tw, text_y)


class EmbroideryViewerPanel(wx.Panel):
    """Fast interactive embroidery preview with playback and viewport controls.

    Stitch data is kept in a NumPy array and rendered into a bitmap by the
    Numba rasterizers above. This panel owns the viewer state: loaded design,
    current stitch position, zoom and pan, grid visibility, playback, and
    keyboard/mouse interaction.
    """

    def __init__(self, parent, progress_bar):
        """Create an empty viewer connected to the progress bar."""
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.zoom = 1.0
        self.pan_x, self.pan_y = 400, 300
        self.drag_start = None
        self.pan_start = (0, 0)
        self.line_width = 0.4
        self.dark_factor = 0.75
        self.light_factor = 0.45
        self.shading_step = 0.05
        self.visible_count = 0
        self.step_size = 10
        self.show_grid = True
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.bounds = (0, 0, 0, 0)
        self.color_boundaries = []
        self.cached_bitmap = None
        self.need_redraw = True
        self.progress_bar = progress_bar
        self._last_key_time = 0
        self._key_throttle = 0.03
        self._last_dir = 1
        self._pending_fit_to_screen = False
        self.play_timer = wx.Timer(self)
        self.play_speed = 20
        self.play_speed_levels = (1, 5, 10, 20, 40, 80)
        self.play_speed_index = 2
        self.play_step = self.play_speed_levels[self.play_speed_index]
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
        """Invalidate the bitmap and retry deferred initial fitting."""
        self.need_redraw = True
        if self._pending_fit_to_screen and self.stitches_np.shape[0] > 0:
            wx.CallAfter(self._try_fit_to_screen)
        e.Skip()

    def _try_fit_to_screen(self, retries=20):
        """Fit the design once wx has assigned a usable panel size."""
        if not self._pending_fit_to_screen:
            return
        w, h = self.GetSize()
        # On startup wx can briefly report tiny panel sizes.
        # If we fit at that moment, the design appears tiny in the top-left.
        # Retry shortly until layout stabilizes.
        if w < 120 or h < 120:
            if retries > 0:
                wx.CallLater(30, self._try_fit_to_screen, retries - 1)
            return
        self._pending_fit_to_screen = False
        self.FitToScreen()

    def FitToScreen(self):
        """Center the loaded design and scale it to fit the viewport."""
        if self.stitches_np.shape[0] == 0:
            return
        min_x, min_y, max_x, max_y = self.bounds
        bw = max_x - min_x
        bh = max_y - min_y
        bw = max(bw, 1)
        bh = max(bh, 1)
        w, h = self.GetSize()
        if w < 10 or h < 10:
            w, h = 1200, 800
        zoom_x = (w * 0.8) / bw
        zoom_y = (h * 0.8) / bh
        self.zoom = min(zoom_x, zoom_y)
        self.CenterDesign()

    def CenterDesign(self):
        """Center the loaded design without changing its current zoom."""
        if self.stitches_np.shape[0] == 0:
            return
        w, h = self.GetSize()
        if w < 10 or h < 10:
            w, h = 1200, 800
        min_x, min_y, max_x, max_y = self.bounds
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        self.pan_x = w / 2 - cx * self.zoom
        self.pan_y = h / 2 - cy * self.zoom
        self.need_redraw = True
        self.Refresh()
        if self.progress_bar:
            self.progress_bar.Refresh()

    def OnPlayTimer(self, e):
        """Advance playback by one timer step in the current direction."""
        total = self.stitches_np.shape[0]
        if total == 0:
            self.play_timer.Stop()
            self.is_playing = False
            return
        self.visible_count += self.play_step * self._last_dir
        if self.visible_count >= total:
            self.visible_count = total
            self.play_timer.Stop()
            self.is_playing = False
        elif self.visible_count <= 0:
            self.visible_count = 0
            self.play_timer.Stop()
            self.is_playing = False
        self.need_redraw = True
        self.Refresh()
        if self.progress_bar:
            self.progress_bar.Refresh()

    def ToggleAutoPlay(self, forward=True):
        """Start or stop playback, choosing its direction when starting."""
        if self.is_playing:
            self.play_timer.Stop()
            self.is_playing = False
        else:
            self._last_dir = 1 if forward else -1
            self.play_timer.Start(self.play_speed)
            self.is_playing = True

    def AdjustPlaybackSpeed(self, direction):
        """Increase or decrease playback speed while preserving its direction."""
        new_index = max(
            0,
            min(len(self.play_speed_levels) - 1, self.play_speed_index + direction),
        )
        if new_index == self.play_speed_index:
            return False
        self.play_speed_index = new_index
        self.play_step = self.play_speed_levels[new_index]
        return True

    def OnKeyUp(self, e):
        """Reset key-repeat throttling after a key is released."""
        self._last_key_time = 0
        e.Skip()

    def JumpToColor(self, direction):
        """Move to the next or previous recorded thread-color boundary."""
        if not self.color_boundaries:
            return
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
                    self.visible_count = self.color_boundaries[idx - 1]
                else:
                    self.visible_count = 0
            else:
                self.visible_count = prev

    def OnKeyDown(self, e):
        """Handle playback, navigation, display, and view shortcut keys."""
        now = time.time()
        key = e.GetKeyCode()
        is_space_or_c = key in (
            wx.WXK_SPACE,
            ord("C"),
            ord("c"),
        )
        if (
            not is_space_or_c
            and now - self._last_key_time < self._key_throttle
            and not e.AltDown()
            and not e.ControlDown()
        ):
            return
        self._last_key_time = now
        total = self.stitches_np.shape[0]
        is_alt = e.AltDown()
        is_ctrl = e.ControlDown()
        changed = False
        step = 1 if is_alt else self.step_size
        if self.is_playing and not is_alt and not is_ctrl and key in (
            wx.WXK_RIGHT,
            wx.WXK_LEFT,
        ):
            key_direction = 1 if key == wx.WXK_RIGHT else -1
            changed = self.AdjustPlaybackSpeed(
                key_direction * self._last_dir
            )
        elif is_ctrl and key in (wx.WXK_RIGHT, wx.WXK_LEFT):
            if key == wx.WXK_RIGHT:
                self.JumpToColor(1)
                self._last_dir = 1
            else:
                self.JumpToColor(-1)
                self._last_dir = -1
            changed = True
        elif key == wx.WXK_RIGHT:
            if self.visible_count < total:
                self.visible_count = min(total, self.visible_count + step)
                self._last_dir = 1
                changed = True
        elif key == wx.WXK_LEFT:
            if self.visible_count > 0:
                self.visible_count = max(0, self.visible_count - step)
                self._last_dir = -1
                changed = True
        elif key == wx.WXK_UP:
            self.visible_count = min(total, self.visible_count + step * 10)
            self._last_dir = 1
            changed = True
        elif key == wx.WXK_DOWN:
            self.visible_count = max(0, self.visible_count - step * 10)
            self._last_dir = -1
            changed = True
        elif key == wx.WXK_HOME:
            self.visible_count = 0
            changed = True
        elif key == wx.WXK_END:
            self.visible_count = total
            changed = True
        elif key == wx.WXK_SPACE:
            self.ToggleAutoPlay(forward=self._last_dir > 0)
            return
        elif key in (ord("+"), ord("="), wx.WXK_NUMPAD_ADD):
            self.line_width = min(1.0, self.line_width + 0.1)
            changed = True
        elif key in (ord("-"), ord("_"), wx.WXK_NUMPAD_SUBTRACT):
            self.line_width = max(0.1, self.line_width - 0.1)
            changed = True
        elif key in (ord("["), ord("{"), ord("]"), ord("}")):
            shading_delta = self.shading_step
            if key in (ord("["), ord("{")):
                shading_delta = -shading_delta
            if e.ShiftDown() or key in (ord("{"), ord("}")):
                self.light_factor = max(
                    0.0,
                    min(1.0, self.light_factor + shading_delta),
                )
            else:
                self.dark_factor = max(
                    0.05,
                    min(1.0, self.dark_factor + shading_delta),
                )
            changed = True
        elif key in (ord("C"), ord("c")):
            self.CenterDesign()
            return
        elif key in (ord("F"), ord("f")):
            self.FitToScreen()
            return
        elif key == wx.WXK_F11:
            frame = wx.GetTopLevelParent(self)
            if hasattr(frame, "ToggleFullScreen"):
                frame.ToggleFullScreen()
                return
        elif key in (ord("G"), ord("g")):
            self.show_grid = not self.show_grid
            changed = True
        elif key in (ord("H"), ord("h")):
            self.ShowHelp()
            return
        elif key in (ord("I"), ord("i")):
            self.ShowSettings()
            return
        elif key == wx.WXK_ESCAPE:
            if self.is_playing:
                self.play_timer.Stop()
                self.is_playing = False
                return
        if changed:
            if (
                self.is_playing
                and key in (wx.WXK_UP, wx.WXK_DOWN, wx.WXK_HOME, wx.WXK_END)
                and not is_ctrl
            ):
                self.play_timer.Stop()
                self.is_playing = False
            self.need_redraw = True
            self.Refresh()
            if self.progress_bar:
                self.progress_bar.Refresh()
        else:
            e.Skip()

    def ShowHelp(self):
        """Show the keyboard and mouse controls used by the viewer."""
        help_text = f"{APP_TITLE}\n\nMouse: Wheel=Zoom Drag=Pan Click bar=Seek\n\nPlayback:\n  Right/Left - speed up/down while playing\n  Alt+Right/Left - +/- 1 stitch when stopped\n  Ctrl+Right/Left - Next/Prev color\n  Up/Down - Fast seek when stopped\n  Home/End - First/Last\n  Space - Play/Pause toggle\n  C - Center design\n  F - Fit design to window\n  F11 - Toggle fullscreen\n  Esc - Stop\n\nView: +/- width G=grid H=help\nShading: [ ] - dark factor  Shift+[ ] - light factor\nInfo: I - viewer settings\n"
        dlg = wx.MessageDialog(self, help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def ShowSettings(self):
        """Show the current viewer state and rendering parameters."""
        total = self.stitches_np.shape[0]
        min_x, min_y, max_x, max_y = self.bounds
        width = max_x - min_x
        height = max_y - min_y
        settings_text = (
            f"{APP_TITLE} viewer settings\n\n"
            f"Design\n"
            f"  Stitches: {self.visible_count}/{total}\n"
            f"  Colors: {len(self.color_boundaries)} boundaries\n"
            f"  Bounds: {width:.1f} x {height:.1f} mm\n\n"
            f"Viewport\n"
            f"  Zoom: {self.zoom:.3f}\n"
            f"  Pan: {self.pan_x:.1f}, {self.pan_y:.1f} px\n"
            f"  Grid: {'on' if self.show_grid else 'off'}\n"
            f"  Gradient: {'on' if self.zoom > 1.2 else 'off'}\n\n"
            f"Rendering\n"
            f"  Line width: {self.line_width:.1f} px\n"
            f"  Dark factor: {self.dark_factor:.2f}\n"
            f"  Light factor: {self.light_factor:.2f}\n"
            f"  Shading step: {self.shading_step:.2f}\n\n"
            f"Playback\n"
            f"  Step size: {self.step_size}\n"
            f"  Timer interval: {self.play_speed} ms\n"
            f"  Timer step: {self.play_step} stitches\n"
            f"  Direction: {'forward' if self._last_dir > 0 else 'backward'}\n"
            f"  Playing: {'yes' if self.is_playing else 'no'}"
        )
        dlg = wx.MessageDialog(
            self,
            settings_text,
            "Viewer settings",
            wx.OK | wx.ICON_INFORMATION,
        )
        dlg.ShowModal()
        dlg.Destroy()

    def SetStepSize(self, size):
        self.step_size = max(1, size)

    def LoadDesign(self, path, fit_to_screen=True):
        """Load an embroidery file into renderable stitch segments."""
        try:
            pattern = emb.read(path)
        except Exception as ex:
            wx.MessageBox(f"Failed to load embroidery file: {ex}", "Error")
            return False
        segs = []
        last_x = last_y = 0
        cur_color_idx = 0
        palette = (
            pattern.threadlist
            if hasattr(pattern, "threadlist") and pattern.threadlist
            else [(220, 30, 30)]
        )
        min_x = min_y = 1e9
        max_x = max_y = -1e9
        self.color_boundaries = [0]
        for st in pattern.stitches:
            x = st[0] / 10.0
            y = st[1] / 10.0
            cmd = st[2] if len(st) > 2 else 0
            if hasattr(emb, "JUMP") and cmd == emb.JUMP:
                last_x, last_y = x, y
                continue
            if hasattr(emb, "COLOR_CHANGE") and (
                cmd == emb.COLOR_CHANGE or (cmd & 0x04)
            ):
                cur_color_idx += 1
                if segs:
                    self.color_boundaries.append(len(segs))
                last_x, last_y = x, y
                continue
            if hasattr(emb, "END") and cmd == emb.END:
                break
            if cur_color_idx < len(palette):
                col = palette[cur_color_idx]
                if hasattr(col, "get_red"):
                    rgb = (col.get_red(), col.get_green(), col.get_blue())
                elif isinstance(col, (list, tuple)):
                    rgb = tuple(col[:3])
                else:
                    rgb = (220, 30, 30)
            else:
                rgb = (220, 30, 30)
            segs.append((last_x, last_y, x, y, rgb[0], rgb[1], rgb[2]))
            min_x = min(min_x, last_x, x)
            min_y = min(min_y, last_y, y)
            max_x = max(max_x, last_x, x)
            max_y = max(max_y, last_y, y)
            last_x, last_y = x, y
        if segs:
            self.stitches_np = np.array(segs, dtype=np.float32)
            self.bounds = (min_x, min_y, max_x, max_y)
            self.visible_count = self.stitches_np.shape[0]
            self.color_boundaries = sorted(set(self.color_boundaries))
            if fit_to_screen:
                self._pending_fit_to_screen = True
                wx.CallAfter(self._try_fit_to_screen)
        self.need_redraw = True
        self.Refresh()
        if self.progress_bar:
            self.progress_bar.Refresh()
        self.SetFocus()
        return True

    def OnPaint(self, e):
        """Render the current viewport, using the cached bitmap when possible."""
        dc = wx.PaintDC(self)
        dc.Clear()
        if not self.need_redraw and self.cached_bitmap:
            dc.DrawBitmap(self.cached_bitmap, 0, 0)
            return
        w, h = self.GetSize()
        if self.stitches_np.shape[0] == 0:
            dc.SetFont(
                wx.Font(
                    14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
                )
            )
            dc.DrawText(
                "Open an embroidery file via File > Open or pass it as a command-line argument",
                20,
                20,
            )
            dc.DrawText(
                "H=help, Space=play/pause, Ctrl+Arrows=color, Alt+Arrows=1", 20, 45
            )
            return
        use_gradient = self.zoom > 1.2
        buf = np.full((h, w, 3), 255, dtype=np.uint8)
        if self.show_grid:
            render_grid_numba(buf, self.zoom, self.pan_x, self.pan_y)
        if self.stitches_np.shape[0] > 0 and self.visible_count > 0:
            render_shaded_numba(
                buf,
                self.stitches_np,
                self.visible_count,
                self.zoom,
                self.pan_x,
                self.pan_y,
                use_gradient,
                self.line_width,
                self.dark_factor,
                self.light_factor,
            )
        img = wx.Image(w, h)
        img.SetData(buf.tobytes())
        bmp = wx.Bitmap(img)
        self.cached_bitmap = bmp
        self.need_redraw = False
        dc.DrawBitmap(bmp, 0, 0)

    def OnWheel(self, e):
        """Zoom around the mouse position while preserving its world point."""
        mx, my = e.GetPosition()
        old = self.zoom
        self.zoom *= 1.15 if e.GetWheelRotation() > 0 else 1 / 1.15
        self.zoom = max(0.05, min(50.0, self.zoom))
        scale = self.zoom / old
        self.pan_x = mx - scale * (mx - self.pan_x)
        self.pan_y = my - scale * (my - self.pan_y)
        self.need_redraw = True
        self.Refresh()

    def OnLeftDown(self, e):
        """Start panning from the current mouse position."""
        self.drag_start = e.GetPosition()
        self.pan_start = (self.pan_x, self.pan_y)
        self.CaptureMouse()
        self.SetFocus()

    def OnLeftUp(self, e):
        """Stop panning and clean up any progress-bar mouse capture."""
        if self.HasCapture():
            self.ReleaseMouse()
        self.drag_start = None
        if self.progress_bar and self.progress_bar.dragging:
            self.progress_bar.dragging=False
            if self.progress_bar.HasCapture(): self.progress_bar.ReleaseMouse()

    def OnMotion(self,e):
        """Update the viewport offset while the user drags the canvas."""
        if self.drag_start and e.Dragging() and e.LeftIsDown():
            dx = e.GetPosition()[0] - self.drag_start[0]
            dy = e.GetPosition()[1] - self.drag_start[1]
            self.pan_x = self.pan_start[0] + dx
            self.pan_y = self.pan_start[1] + dy
            self.need_redraw = True
            self.Refresh()


class Frame(wx.Frame):
    """Main InkSim window coordinating the viewer and playback controls.

    The initial design is loaded before the frame is shown.  Fullscreen
    startup also gives the frame the display size before loading the design,
    then performs one final fit after wx has completed the layout.  This avoids
    showing an incorrectly positioned design while GTK applies fullscreen
    geometry asynchronously.
    """

    def __init__(
        self,
        initial_file=None,
        fullscreen=False,
        window_size=None,
        window_position=None,
        autoplay=False,
    ):
        """Build the application window and optionally open a design file."""
        # Decide initial size before super().__init__
        # -f: use display size
        # default MaxWindow: also use display size so first FitToScreen is already correct
        # explicit --size: use that size
        init_size = (1200, 980)
        should_maximize_default = False
        if not window_size:
            try:
                disp = wx.Display(0).GetGeometry()
                disp_size = (disp.GetWidth(), disp.GetHeight())
                if fullscreen:
                    init_size = disp_size
                else:
                    # default MaxWindow behavior requested by user
                    init_size = disp_size
                    should_maximize_default = True
            except Exception:
                init_size = (1200, 980)
                should_maximize_default = not fullscreen
        else:
            init_size = window_size

        super().__init__(None, title=APP_TITLE, size=init_size)
        self.is_fullscreen = False
        self._should_maximize_default = should_maximize_default

        # Create the main panel, viewer, and progress bar, and arrange them vertically.
        main_panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.viewer = EmbroideryViewerPanel(main_panel, None)
        self.progress = ProgressBarPanel(main_panel, self.viewer)
        self.viewer.progress_bar = self.progress
        self.progress.Bind(wx.EVT_LEFT_UP, self.viewer.OnLeftUp)

        sizer.Add(self.viewer, 1, wx.EXPAND)
        sizer.Add(self.progress, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
                  6)
        main_panel.SetSizer(sizer)
        self._main_panel = main_panel

        # Build the menu bar with file and playback options, and bind them to handlers.
        menubar = wx.MenuBar()

        fileMenu = wx.Menu()
        openItem = fileMenu.Append(wx.ID_OPEN, "Open embroidery file\tCtrl+O")
        centerItem = fileMenu.Append(wx.ID_ANY, "Center design\tC")
        fitItem = fileMenu.Append(wx.ID_ANY, "Fit design to window\tF")
        fullscreenItem = fileMenu.Append(wx.ID_ANY, "Fullscreen\tF11")
        gridItem = fileMenu.AppendCheckItem(wx.ID_ANY, "Show 1cm grid\tG")
        gridItem.Check(True)
        helpItem = fileMenu.Append(wx.ID_ANY, "Help\tH")
        menubar.Append(fileMenu, "File")

        playbackMenu = wx.Menu()
        s1 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 1 (Alt+Arrows)")
        s10 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 10")
        s10.Check(True)
        s50 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 50")
        s100 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 100")
        s500 = playbackMenu.AppendRadioItem(wx.ID_ANY, "Step 500")
        playbackMenu.AppendSeparator()

        playItem = playbackMenu.Append(wx.ID_ANY, "Play/Pause\tSpace")
        nextCol = playbackMenu.Append(wx.ID_ANY, "Next color\tCtrl+Right")
        prevCol = playbackMenu.Append(wx.ID_ANY, "Prev color\tCtrl+Left")
        menubar.Append(playbackMenu, "Playback")
        self.SetMenuBar(menubar)

        # Bind menu items to their handlers.
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_MENU, self.OnOpen, openItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.CenterDesign(), centerItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.FitToScreen(), fitItem)
        self.Bind(wx.EVT_MENU, lambda e: self.ToggleFullScreen(), fullscreenItem)
        self.Bind(wx.EVT_MENU, self.OnToggleGrid, gridItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.ShowHelp(), helpItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(1), s1)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(10), s10)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(50), s50)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(100), s100)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.SetStepSize(500), s500)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.ToggleAutoPlay(True),
                  playItem)
        self.Bind(
            wx.EVT_MENU, lambda e: self.viewer.JumpToColor(1) or self.
            _refresh_after_color_jump(), nextCol)
        self.Bind(
            wx.EVT_MENU, lambda e: self.viewer.JumpToColor(-1) or self.
            _refresh_after_color_jump(), prevCol)

        # Set up the status bar with instructions.
        self.CreateStatusBar()
        self.SetStatusText(
            "Space=play/pause | C=center | F=fit | F11=fullscreen | Ctrl+Arrows=color | G=grid H=help"
        )

        # Window geometry
        if window_size:
            self.SetSize(window_size)
        if window_position:
            self.SetPosition(window_position)
        elif not window_size and not fullscreen and not should_maximize_default:
            self.Centre()

        # Load design with no auto-fit, we will fit explicitly after final size.
        initial_file_loaded = (
            initial_file
            and os.path.exists(initial_file)
            and self.viewer.LoadDesign(initial_file, fit_to_screen=False)
        )
        if initial_file_loaded:
            total = self.viewer.stitches_np.shape[0]
            self.SetTitle(
                f"{APP_TITLE} - {os.path.basename(initial_file)} - {total} sts"
            )

        if fullscreen:
            self.is_fullscreen = True
            self.Freeze()
            if not self.IsShown():
                self.Show()
            self.ShowFullScreen(True)
            self.Layout()
            self._main_panel.Layout()
            self.viewer.Layout()
            wx.CallAfter(self._finish_initial_display, autoplay)
        elif should_maximize_default:
            # Default MaxWindow - start maximized but without flicker.
            # Size is already display size, so first Fit is already almost correct.
            # Freeze to hide intermediate paint, then Maximize and fit again after GTK event.
            self.Freeze()
            if not self.IsShown():
                self.Show()
            # On GTK Maximize is async, so we need one more layout pass after it.
            self.Maximize(True)
            self.Layout()
            self._main_panel.Layout()
            self.viewer.Layout()
            wx.CallAfter(self._finish_initial_display, autoplay)
        else:
            if not self.IsShown():
                self.Show()
            if initial_file_loaded:
                wx.CallAfter(self._finish_initial_display, autoplay)

    def _finish_initial_display(self, autoplay):
        """Finish the one-time startup layout before playback begins.

        ``wx.CallAfter`` runs this after the frame and child panels have their
        final sizes.  The fit is intentionally limited to startup; changing
        fullscreen later with ``F11`` preserves the user's current viewport.
        """
        self.Layout()
        self._main_panel.Layout()
        self.viewer.Layout()
        # Final fit using real client size, not temporary 1200x980
        self.viewer.FitToScreen()
        if self.IsFrozen():
            self.Thaw()
        self.viewer.need_redraw = True
        self.viewer.Refresh()
        self.progress.Refresh()
        if autoplay:
            self.viewer.visible_count = 0
            self.viewer.need_redraw = True
            self.viewer.Refresh()
            self.progress.Refresh()
            self.viewer.ToggleAutoPlay(forward=True)

    def _refresh_after_color_jump(self):
        """Refresh the viewer and timeline after a color-boundary jump."""
        self.viewer.need_redraw = True
        self.viewer.Refresh()
        self.progress.Refresh()

    def OnClose(self, e):
        """Stop playback before allowing the frame to close."""
        if self.viewer.is_playing:
            self.viewer.play_timer.Stop()
            self.viewer.is_playing = False
        e.Skip()

    def OnToggleGrid(self, e):
        """Apply the grid menu state to the viewer and redraw it."""
        self.viewer.show_grid = e.IsChecked()
        self.viewer.need_redraw = True
        self.viewer.Refresh()

    def OnOpen(self, e):
        """Prompt for an embroidery file and update the window metadata."""
        dlg = wx.FileDialog(self,
                    "Open embroidery file",
                    wildcard=get_supported_input_wildcard(),
                            style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            if self.viewer.LoadDesign(path, fit_to_screen=True):
                total = self.viewer.stitches_np.shape[0]
                bw = self.viewer.bounds[2] - self.viewer.bounds[0]
                bh = self.viewer.bounds[3] - self.viewer.bounds[1]
                self.SetTitle(
                    f"{APP_TITLE} - {os.path.basename(path)} - {total} sts - {bw:.1f}x{bh:.1f}mm"
                )
                self.progress.Refresh()
        dlg.Destroy()

    def ToggleFullScreen(self):
        """Toggle undecorated fullscreen without changing the viewport."""
        self.is_fullscreen = not self.is_fullscreen
        self.ShowFullScreen(self.is_fullscreen)


def _parse_pair(value, name, separator):
    """Parse two integer values used for window geometry."""
    parts = value.split(separator)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"{name} must use the format VALUE{separator}VALUE"
        )
    try:
        first, second = (int(part) for part in parts)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(f"{name} values must be integers") from ex
    if name == "size" and (first <= 0 or second <= 0):
        raise argparse.ArgumentTypeError("size values must be greater than zero")
    return first, second


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("input_file", nargs="?", help="Input embroidery file")
    parser.add_argument(
        "-f", "--fullscreen", action="store_true",
        help="Open the simulator fullscreen",
    )
    parser.add_argument(
        "-p", "--play", action="store_true",
        help="Start simulation playback immediately",
    )
    parser.add_argument(
        "--size",
        metavar="WIDTHxHEIGHT",
        type=lambda value: _parse_pair(value, "size", "x"),
        help="Window size, for example 1600x1000",
    )
    parser.add_argument(
        "--position",
        metavar="X,Y",
        type=lambda value: _parse_pair(value, "position", ","),
        help="Window position, for example 100,50",
    )
    args=parser.parse_args()

    window_size = args.size
    window_position = args.position
    app=wx.App()
    Frame(
        initial_file=args.input_file,
        fullscreen=args.fullscreen,
        window_size=window_size,
        window_position=window_position,
        autoplay=args.play,
    )
    app.MainLoop()
