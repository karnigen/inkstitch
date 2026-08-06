#!/usr/bin/env python3

from pathlib import Path
import sys
import os
import argparse

#-------------------------------------------------------------------
# Dual run: system python3 or .venv python

script_dir = Path(__file__).resolve().parent

APP_TITLE = "InkSim"
DEFAULT_STATUS_TEXT = (
    "Space=play/pause | C=center | F=fit | F11=fullscreen | "
    "Ctrl+Arrows=color | G=grid H=help"
)
DENSITY_RADIUS_MM = 2.5
DENSITY_WARNING_PER_MM2 = 3.0
DENSITY_CRITICAL_PER_MM2 = 6.0
MAX_RENDER_LINE_WIDTH_PX = 16.0
MAX_RENDER_STEPS = 2048
REALISTIC_END_FADE_PX = 4.0
AUTO_THREAD_COLORS = (
    (220, 30, 30),
    (30, 100, 220),
    (30, 160, 80),
    (230, 150, 25),
    (150, 60, 180),
    (20, 170, 180),
    (220, 70, 140),
    (110, 110, 110),
)

def ensure_venv():
    if sys.prefix != sys.base_prefix:
        return

    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv:
        project_root = Path(active_venv).resolve()
    else:
        project_root = next(
            (
                parent
                for parent in (script_dir, *script_dir.parents)
                if (parent / "pyproject.toml").is_file()
            ),
            None,
        )
        if project_root is None:
            return
        project_root = project_root / ".venv"

    if os.name == "nt":
        venv_python = project_root / "Scripts" / "python.exe"
    else:
        venv_python = project_root / "bin" / "python"

    if venv_python.exists():
        os.execv(venv_python, [venv_python] + sys.argv)

# restart the virtual environment if not already active
ensure_venv()
#-------------------------------------------------------------------

import wx
import time
import numpy as np
import numba
import pystitch as emb
from PIL import Image, ImageDraw, PngImagePlugin

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
    use_shaded,
    line_width,
    dark_factor,
    light_factor,
    use_realistic=False,
):
    # Draw visible stitch segments into the RGB buffer.
    h, w, _ = buf.shape
    minimum_line_width = 1.0
    effective_line_width = min(
        MAX_RENDER_LINE_WIDTH_PX,
        max(minimum_line_width, line_width * zoom),
    )
    hw = effective_line_width * 0.5
    lw_int = max(1, int(np.ceil(effective_line_width)))

    light_x = -0.5
    light_y = -0.8660254

    # For realistic: puncture config
    puncture_radius = 1
    if hw > 2.5:
        puncture_radius = 2
    last_puncture_x = -100000
    last_puncture_y = -100000

    for i in range(visible_count):
        x1 = stitches[i,0] * zoom + pan_x
        y1 = stitches[i,1] * zoom + pan_y
        x2 = stitches[i,2] * zoom + pan_x
        y2 = stitches[i,3] * zoom + pan_y

        r_base = int(stitches[i, 4])
        g_base = int(stitches[i, 5])
        b_base = int(stitches[i, 6])

        if (x1 < -200 and x2 < -200) or (x1 > w+200 and x2 > w+200):
            # Still need to handle puncture skipping for invisible stitches
            pass
        elif (y1 < -200 and y2 < -200) or (y1 > h+200 and y2 > h+200):
            pass
        else:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                tangent_x = dx / length
                tangent_y = dy / length
                normal_x = -tangent_y
                normal_y = tangent_x
                if normal_x * light_x + normal_y * light_y < 0.0:
                    normal_x = -normal_x
                    normal_y = -normal_y
                dot_tl = tangent_x * light_x + tangent_y * light_y
                perp_sq = 1.0 - dot_tl * dot_tl
                if perp_sq < 0.0: perp_sq = 0.0
                perp_factor = np.sqrt(perp_sq)

                r_dark = int(r_base * (0.85 + 0.15 * dark_factor))
                g_dark = int(g_base * (0.85 + 0.15 * dark_factor))
                b_dark = int(b_base * (0.85 + 0.15 * dark_factor))
                r_light = int(r_base + (255 - r_base) * min(1.0, light_factor + 0.10))
                g_light = int(g_base + (255 - g_base) * min(1.0, light_factor + 0.10))
                b_light = int(b_base + (255 - b_base) * min(1.0, light_factor + 0.10))
                r_bright = int(min(255, r_base * 1.08 + 12))
                g_bright = int(min(255, g_base * 1.08 + 12))
                b_bright = int(min(255, b_base * 1.08 + 12))

                steps = min(MAX_RENDER_STEPS, max(1, int(np.ceil(length))))
                for s in range(steps+1):
                    t = s / steps
                    x = x1 + dx * t
                    y = y1 + dy * t

                    if use_shaded:
                        r = int(r_dark + (r_light - r_dark) * t)
                        g = int(g_dark + (g_light - g_dark) * t)
                        b = int(b_dark + (b_light - b_dark) * t)
                    else:
                        r = r_base
                        g = g_base
                        b = b_base

                    if lw_int <= 1 and not use_realistic:
                        xi = int(x)
                        yi = int(y)
                        if 0 <= xi < w and 0 <= yi < h:
                            buf[yi, xi, 0] = r
                            buf[yi, xi, 1] = g
                            buf[yi, xi, 2] = b
                    else:
                        render_radius = hw
                        r_loop = lw_int + 1
                        for oy in range(-r_loop, r_loop + 1):
                            for ox in range(-r_loop, r_loop + 1):
                                distance_squared = ox*ox + oy*oy
                                if distance_squared > render_radius*render_radius + 0.5:
                                    continue
                                xi = int(x + ox)
                                yi = int(y + oy)
                                if 0 <= xi < w and 0 <= yi < h:
                                    if use_realistic:
                                        normal_position = ox * normal_x + oy * normal_y
                                        across = normal_position / hw if hw > 0.001 else 0.0
                                        if across < -1.0: across = -1.0
                                        if across >  1.0: across =  1.0
                                        across_abs = across if across >= 0 else -across
                                        cyl = 1.0 - across_abs * across_abs * 0.7
                                        rr = int(r_dark + (r_bright - r_dark) * cyl)
                                        gg = int(g_dark + (g_bright - g_dark) * cyl)
                                        bb = int(b_dark + (b_bright - b_dark) * cyl)

                                        zoom_fade = (hw - 1.2) / 1.3
                                        if zoom_fade < 0.0: zoom_fade = 0.0
                                        if zoom_fade > 1.0: zoom_fade = 1.0
                                        if zoom_fade > 0.01:
                                            spec_center = 0.25
                                            spec_width = 0.50
                                            spec_dist = across - spec_center
                                            if spec_dist < 0: spec_dist = -spec_dist
                                            if spec_dist < spec_width:
                                                spec = 1.0 - spec_dist / spec_width
                                                spec = spec * spec
                                                along = t if t < 0.5 else 1.0 - t
                                                if along < 0.20:
                                                    spec *= along / 0.20
                                                angle_mod = 0.40 + 0.60 * perp_factor
                                                spec *= angle_mod * zoom_fade
                                                spec_strength = spec * 0.32
                                                hl_r = int((rr + r_light + 255) * 0.33 + 10)
                                                hl_g = int((gg + g_light + 255) * 0.33 + 10)
                                                hl_b = int((bb + b_light + 255) * 0.33 + 10)
                                                if hl_r > 255: hl_r = 255
                                                if hl_g > 255: hl_g = 255
                                                if hl_b > 255: hl_b = 255
                                                rr = int(rr * (1.0 - spec_strength) + hl_r * spec_strength)
                                                gg = int(gg * (1.0 - spec_strength) + hl_g * spec_strength)
                                                bb = int(bb * (1.0 - spec_strength) + hl_b * spec_strength)

                                        if distance_squared > (hw - 0.7)*(hw - 0.7):
                                            buf[yi, xi, 0] = (buf[yi, xi, 0] * 2 + rr) // 3
                                            buf[yi, xi, 1] = (buf[yi, xi, 1] * 2 + gg) // 3
                                            buf[yi, xi, 2] = (buf[yi, xi, 2] * 2 + bb) // 3
                                        else:
                                            if across_abs < 0.6:
                                                buf[yi, xi, 0] = rr
                                                buf[yi, xi, 1] = gg
                                                buf[yi, xi, 2] = bb
                                            else:
                                                old_lum = buf[yi, xi, 0] + buf[yi, xi, 1] + buf[yi, xi, 2]
                                                new_lum = rr + gg + bb
                                                if new_lum >= old_lum - 30:
                                                    buf[yi, xi, 0] = rr
                                                    buf[yi, xi, 1] = gg
                                                    buf[yi, xi, 2] = bb
                                    elif distance_squared <= (hw-0.5)*(hw-0.5):
                                        buf[yi, xi, 0] = r
                                        buf[yi, xi, 1] = g
                                        buf[yi, xi, 2] = b
                                    else:
                                        buf[yi, xi, 0] = (buf[yi, xi, 0] + r)//2
                                        buf[yi, xi, 1] = (buf[yi, xi, 1] + g)//2
                                        buf[yi, xi, 2] = (buf[yi, xi, 2] + b)//2

        # ---- Puncture handling: draw puncture from previous stitch AFTER current stitch ----
        # This way puncture is on top of both stitches sharing it, but will be naturally
        # overdrawn by any later stitch that crosses it (including top layer over underlay).
        # No lookahead needed - natural overdraw does the occlusion.
        if use_realistic and effective_line_width > 1.8 and i > 0:
            # Puncture position is end of previous stitch (junction)
            # Use i-1 stitch end
            px = int(stitches[i-1,2] * zoom + pan_x)
            py = int(stitches[i-1,3] * zoom + pan_y)
            if px >= 0 and px < w and py >= 0 and py < h:
                # Skip if too close to last drawn puncture (dense satin)
                ddx = px - last_puncture_x
                ddy = py - last_puncture_y
                if ddx*ddx + ddy*ddy >= 4:
                    r_pb = int(stitches[i-1, 4])
                    g_pb = int(stitches[i-1, 5])
                    b_pb = int(stitches[i-1, 6])
                    r_hole = int(r_pb * 0.45)
                    g_hole = int(g_pb * 0.45)
                    b_hole = int(b_pb * 0.45)
                    r_ring = int(r_pb * 0.75)
                    g_ring = int(g_pb * 0.75)
                    b_ring = int(b_pb * 0.75)

                    for oy in range(-puncture_radius, puncture_radius+1):
                        for ox in range(-puncture_radius, puncture_radius+1):
                            xi = px + ox
                            yi = py + oy
                            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                                continue
                            dist2 = ox*ox + oy*oy
                            if dist2 == 0:
                                buf[yi, xi, 0] = r_hole
                                buf[yi, xi, 1] = g_hole
                                buf[yi, xi, 2] = b_hole
                            elif dist2 == 1:
                                buf[yi, xi, 0] = (buf[yi, xi, 0] * 2 + r_ring) // 3
                                buf[yi, xi, 1] = (buf[yi, xi, 1] * 2 + g_ring) // 3
                                buf[yi, xi, 2] = (buf[yi, xi, 2] * 2 + b_ring) // 3
                            elif dist2 == 2 and puncture_radius > 1:
                                buf[yi, xi, 0] = (buf[yi, xi, 0] * 3 + r_ring) // 4
                                buf[yi, xi, 1] = (buf[yi, xi, 1] * 3 + g_ring) // 4
                                buf[yi, xi, 2] = (buf[yi, xi, 2] * 3 + b_ring) // 4
                    last_puncture_x = px
                    last_puncture_y = py

    # Final puncture at the very end of design
    if use_realistic and effective_line_width > 1.8 and visible_count > 0:
        px = int(stitches[visible_count-1,2] * zoom + pan_x)
        py = int(stitches[visible_count-1,3] * zoom + pan_y)
        if px >= 0 and px < w and py >= 0 and py < h:
            ddx = px - last_puncture_x
            ddy = py - last_puncture_y
            if ddx*ddx + ddy*ddy >= 4:
                r_pb = int(stitches[visible_count-1, 4])
                g_pb = int(stitches[visible_count-1, 5])
                b_pb = int(stitches[visible_count-1, 6])
                r_hole = int(r_pb * 0.45)
                g_hole = int(g_pb * 0.45)
                b_hole = int(b_pb * 0.45)
                r_ring = int(r_pb * 0.75)
                g_ring = int(g_pb * 0.75)
                b_ring = int(b_pb * 0.75)
                for oy in range(-puncture_radius, puncture_radius+1):
                    for ox in range(-puncture_radius, puncture_radius+1):
                        xi = px + ox
                        yi = py + oy
                        if xi < 0 or xi >= w or yi < 0 or yi >= h:
                            continue
                        dist2 = ox*ox + oy*oy
                        if dist2 == 0:
                            buf[yi, xi, 0] = r_hole
                            buf[yi, xi, 1] = g_hole
                            buf[yi, xi, 2] = b_hole
                        elif dist2 == 1:
                            buf[yi, xi, 0] = (buf[yi, xi, 0] * 2 + r_ring) // 3
                            buf[yi, xi, 1] = (buf[yi, xi, 1] * 2 + g_ring) // 3
                            buf[yi, xi, 2] = (buf[yi, xi, 2] * 2 + b_ring) // 3
                        elif dist2 == 2 and puncture_radius > 1:
                            buf[yi, xi, 0] = (buf[yi, xi, 0] * 3 + r_ring) // 4
                            buf[yi, xi, 1] = (buf[yi, xi, 1] * 3 + g_ring) // 4
                            buf[yi, xi, 2] = (buf[yi, xi, 2] * 3 + b_ring) // 4


@numba.njit
def calculate_stitch_density_numba(points, min_x, min_y, max_x, max_y):
    """Calculate stitch endpoints per square millimeter in a 5 mm circle."""
    point_count = points.shape[0]
    density = np.zeros(point_count, dtype=np.float32)
    if point_count == 0:
        return density

    cell_size = 1.0
    grid_width = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    grid_height = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)
    grid = np.zeros((grid_height, grid_width), dtype=np.int32)

    for point_index in range(point_count):
        cell_x = int((points[point_index, 0] - min_x) / cell_size)
        cell_y = int((points[point_index, 1] - min_y) / cell_size)
        cell_x = min(max(cell_x, 0), grid_width - 1)
        cell_y = min(max(cell_y, 0), grid_height - 1)
        grid[cell_y, cell_x] += 1

    radius = DENSITY_RADIUS_MM
    radius_cells = int(np.ceil(radius / cell_size))
    radius_squared = radius * radius
    circle_area = np.pi * radius_squared
    for point_index in range(point_count):
        cell_x = int((points[point_index, 0] - min_x) / cell_size)
        cell_y = int((points[point_index, 1] - min_y) / cell_size)
        count = 0
        for offset_y in range(-radius_cells, radius_cells + 1):
            neighbor_y = cell_y + offset_y
            if neighbor_y < 0 or neighbor_y >= grid_height:
                continue
            for offset_x in range(-radius_cells, radius_cells + 1):
                neighbor_x = cell_x + offset_x
                if neighbor_x < 0 or neighbor_x >= grid_width:
                    continue
                cell_center_x = min_x + (neighbor_x + 0.5) * cell_size
                cell_center_y = min_y + (neighbor_y + 0.5) * cell_size
                dx = cell_center_x - points[point_index, 0]
                dy = cell_center_y - points[point_index, 1]
                if dx * dx + dy * dy <= radius_squared + 1.0:
                    count += grid[neighbor_y, neighbor_x]
        density[point_index] = count / circle_area
    return density


@numba.njit
def render_density_numba(buf, points, density, visible_count, zoom, pan_x, pan_y):
    """Render the stitch-density map directly into the RGB buffer."""
    height, width, _ = buf.shape
    visible_points = min(visible_count, points.shape[0])
    for point_index in range(visible_points):
        density_value = density[point_index]
        if density_value >= DENSITY_CRITICAL_PER_MM2:
            r, g, b = 220, 35, 35
        elif density_value >= DENSITY_WARNING_PER_MM2:
            r, g, b = 235, 175, 25
        else:
            r, g, b = 45, 110, 215
        screen_x = int(points[point_index, 0] * zoom + pan_x)
        screen_y = int(points[point_index, 1] * zoom + pan_y)
        if screen_x < -3 or screen_x >= width + 3:
            continue
        if screen_y < -3 or screen_y >= height + 3:
            continue
        for offset_y in range(-3, 4):
            for offset_x in range(-3, 4):
                if offset_x * offset_x + offset_y * offset_y > 9:
                    continue
                pixel_x = screen_x + offset_x
                pixel_y = screen_y + offset_y
                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                    if offset_x * offset_x + offset_y * offset_y <= 1:
                        buf[pixel_y, pixel_x, 0] = 10
                        buf[pixel_y, pixel_x, 1] = 10
                        buf[pixel_y, pixel_x, 2] = 10
                    else:
                        buf[pixel_y, pixel_x, 0] = r
                        buf[pixel_y, pixel_x, 1] = g
                        buf[pixel_y, pixel_x, 2] = b


def get_supported_input_wildcard():
    """Build a wx file filter from the formats readable by pystitch."""
    extensions = get_supported_input_extensions()
    patterns = ";".join(f"*.{ext}" for ext in sorted(extensions))
    return f"Embroidery files ({patterns})|{patterns}|All files|*.*"


def get_supported_input_extensions():
    """Return lowercase filename extensions readable by pystitch."""
    extensions = set()
    for file_type in emb.EmbPattern.supported_formats():
        if file_type.get("reader") is None:
            continue
        file_extensions = file_type.get("extensions", ())
        if isinstance(file_extensions, str):
            file_extensions = (file_extensions,)
        extensions.update(ext.lstrip(".").lower() for ext in file_extensions)
    return extensions


def render_export_image(stitches, bounds, width, height, line_width, dpi=None,
                        background="transparent", grid=False, shaded=False,
                        dark_factor=0.75, light_factor=0.45):
    """Render clean embroidery geometry into a standalone RGBA PNG image."""
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if background == "white":
        draw.rectangle((0, 0, width, height), fill=(255, 255, 255, 255))

    min_x, min_y, max_x, max_y = bounds
    design_width = max(max_x - min_x, 1.0)
    design_height = max(max_y - min_y, 1.0)
    margin = max(12, min(width, height) * 0.06)
    zoom = min(
        (width - 2 * margin) / design_width,
        (height - 2 * margin) / design_height,
    )
    offset_x = (width - design_width * zoom) / 2 - min_x * zoom
    offset_y = (height - design_height * zoom) / 2 - min_y * zoom

    if grid:
        grid_color = (205, 205, 205, 150)
        for grid_x in range(int(np.floor(min_x / 10)) * 10,
                           int(np.ceil(max_x / 10)) * 10 + 1, 10):
            x = int(grid_x * zoom + offset_x)
            if 0 <= x < width:
                draw.line((x, 0, x, height), fill=grid_color, width=1)
        for grid_y in range(int(np.floor(min_y / 10)) * 10,
                           int(np.ceil(max_y / 10)) * 10 + 1, 10):
            y = int(grid_y * zoom + offset_y)
            if 0 <= y < height:
                draw.line((0, y, width, y), fill=grid_color, width=1)

    stroke_width = max(1, round(max(line_width, 0.7) * zoom))
    for x1, y1, x2, y2, red, green, blue in stitches:
        start = (x1 * zoom + offset_x, y1 * zoom + offset_y)
        end = (x2 * zoom + offset_x, y2 * zoom + offset_y)
        if not shaded:
            draw.line(
                (round(start[0]), round(start[1]), round(end[0]), round(end[1])),
                fill=(int(red), int(green), int(blue), 255),
                width=stroke_width,
            )
            continue
        dark = (
            int(red * dark_factor),
            int(green * dark_factor),
            int(blue * dark_factor),
        )
        light = (
            int(red + (255 - red) * light_factor),
            int(green + (255 - green) * light_factor),
            int(blue + (255 - blue) * light_factor),
        )
        for sample in range(4):
            start_ratio = sample / 4
            end_ratio = (sample + 1) / 4
            start_point = (
                round(start[0] + (end[0] - start[0]) * start_ratio),
                round(start[1] + (end[1] - start[1]) * start_ratio),
            )
            end_point = (
                round(start[0] + (end[0] - start[0]) * end_ratio),
                round(start[1] + (end[1] - start[1]) * end_ratio),
            )
            ratio = (start_ratio + end_ratio) / 2
            color = tuple(
                int(dark[channel] + (light[channel] - dark[channel]) * ratio)
                for channel in range(3)
            )
            draw.line(
                (*start_point, *end_point),
                fill=(*color, 255),
                width=stroke_width,
            )

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("InkSim design size", f"{design_width:.2f} x {design_height:.2f} mm")
    metadata.add_text("InkSim background", background)
    metadata.add_text("InkSim layers", "embroidery only")
    metadata.add_text("InkSim rendering", "shaded" if shaded else "flat")
    if dpi:
        metadata.add_text("InkSim DPI", str(dpi))
    return image, metadata


class EmbroideryFileDropTarget(wx.FileDropTarget):
    """Open the first dropped file in the owning frame."""

    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def OnDropFiles(self, x, y, filenames):
        if filenames:
            self.frame.OpenFile(filenames[0])
        return True


class EmbroideryOpenDialog(wx.Dialog):
    """Browse embroidery files with an in-app design preview."""

    def __init__(self, parent, initial_directory, selected_file=None):
        super().__init__(parent, title="Open embroidery file", size=(1100, 720))
        self.selected_path = None
        self.current_directory = Path(initial_directory or Path.cwd()).resolve()
        self.initial_file = Path(selected_file).resolve() if selected_file else None
        self.extensions = get_supported_input_extensions()

        root_sizer = wx.BoxSizer(wx.VERTICAL)
        directory_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.directory_text = wx.ComboBox(
            self,
            value=str(self.current_directory),
            style=wx.TE_PROCESS_ENTER,
        )
        directory_sizer.Add(self.directory_text, 1, wx.EXPAND | wx.RIGHT, 6)
        up_button = wx.Button(self, label="Up")
        browse_button = wx.Button(self, label="Browse...")
        directory_sizer.Add(up_button, 0, wx.RIGHT, 6)
        directory_sizer.Add(browse_button, 0)
        root_sizer.Add(directory_sizer, 0, wx.EXPAND | wx.ALL, 8)

        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.file_list = wx.ListBox(self)
        content_sizer.Add(self.file_list, 0, wx.EXPAND | wx.LEFT | wx.BOTTOM, 8)
        preview_container = wx.Panel(self)
        preview_sizer = wx.BoxSizer(wx.VERTICAL)
        self.preview = EmbroideryViewerPanel(preview_container, None)
        self.preview.show_grid = False
        self.preview.show_needle = False
        preview_sizer.Add(self.preview, 1, wx.EXPAND)
        preview_container.SetSizer(preview_sizer)
        content_sizer.Add(preview_container, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root_sizer.Add(content_sizer, 1, wx.EXPAND)

        button_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        root_sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizer(root_sizer)

        self.file_list.Bind(wx.EVT_LISTBOX, self.OnSelect)
        self.file_list.Bind(wx.EVT_LISTBOX_DCLICK, self.OnOpen)
        self.directory_text.Bind(wx.EVT_TEXT_ENTER, self.OnDirectoryEnter)
        self.directory_text.Bind(wx.EVT_COMBOBOX, self.OnDirectoryEnter)
        up_button.Bind(wx.EVT_BUTTON, self.OnUp)
        browse_button.Bind(wx.EVT_BUTTON, self.OnBrowse)
        self.Bind(wx.EVT_BUTTON, self.OnOpen, id=wx.ID_OK)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, id=wx.ID_CANCEL)

        self.RefreshFiles()

    def RefreshFiles(self):
        """Refresh the list for the current directory."""
        if not self.current_directory.is_dir():
            return
        directories = sorted(
            (path for path in self.current_directory.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        )
        directory_choices = [str(self.current_directory), *(str(path) for path in directories)]
        self.directory_text.SetItems(directory_choices)
        self.directory_text.SetValue(str(self.current_directory))
        files = sorted(
            (
                path for path in self.current_directory.iterdir()
                if path.is_file() and path.suffix.lower().lstrip(".") in self.extensions
            ),
            key=lambda path: path.name.lower(),
        )
        self.file_list.Set([path.name for path in files])
        self.file_paths = files
        if files:
            selected_index = 0
            if self.initial_file:
                for index, path in enumerate(files):
                    if path == self.initial_file:
                        selected_index = index
                        break
            self.file_list.SetSelection(selected_index)
            self.OnSelect(None)

    def OnSelect(self, event):
        """Load the selected file into the preview panel."""
        selection = self.file_list.GetSelection()
        if selection == wx.NOT_FOUND:
            return
        self.selected_path = self.file_paths[selection]
        self.preview.LoadDesign(str(self.selected_path), fit_to_screen=True)

    def OnOpen(self, event):
        """Accept the selected file."""
        if self.selected_path:
            self.EndModal(wx.ID_OK)

    def OnCancel(self, event):
        self.EndModal(wx.ID_CANCEL)

    def OnDirectoryEnter(self, event):
        self.SetDirectory(self.directory_text.GetValue())

    def SetDirectory(self, directory):
        """Change directory if it exists."""
        path = Path(directory).expanduser().resolve()
        if path.is_dir() and path != self.current_directory:
            self.current_directory = path
            self.initial_file = None
            self.RefreshFiles()

    def OnUp(self, event):
        self.SetDirectory(self.current_directory.parent)

    def OnBrowse(self, event):
        dialog = wx.DirDialog(
            self,
            "Choose directory",
            str(self.current_directory),
        )
        try:
            if dialog.ShowModal() == wx.ID_OK:
                self.SetDirectory(dialog.GetPath())
        finally:
            dialog.Destroy()

    def GetPath(self):
        return str(self.selected_path) if self.selected_path else ""


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
        self.drag_moved = False
        self.margin_x = 24
        self.bar_y = 8
        self.bar_h = 14

    def OnClick(self, e):
        """Start seeking at the mouse position."""
        self.dragging = True
        self.drag_moved = False
        self.Seek(e.GetPosition().x)
        self.viewer.HighlightNeedle()
        self.CaptureMouse()

    def OnLeftUp(self, e):
        """Finish a seek operation and release the mouse capture."""
        if self.dragging:
            self.Seek(e.GetPosition().x)
            self.viewer.HighlightNeedle()
            if self.HasCapture():
                self.ReleaseMouse()
            self.dragging = False
            self.drag_moved = False

    def OnMotionClick(self, e):
        """Update the seek position while the left button is dragged."""
        if self.dragging and e.Dragging() and e.LeftIsDown():
            self.drag_moved = True
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

        command_colors = {
            "JUMP": wx.Colour(100, 100, 100),
            "COLOR CHANGE": wx.Colour(210, 45, 45),
            "TRIM": wx.Colour(230, 140, 20),
            "STOP": wx.Colour(180, 40, 40),
            "SLOW": wx.Colour(70, 100, 180),
            "FAST": wx.Colour(40, 150, 90),
        }
        for stitch_index, commands in self.viewer.command_events.items():
            marker_x = bar_x + int((stitch_index / total) * bar_w)
            for marker_index, command in enumerate(commands):
                marker_y = bar_y + marker_index * 5
                marker = [
                    (marker_x, marker_y),
                    (marker_x - 4, marker_y + 5),
                    (marker_x + 4, marker_y + 5),
                ]
                color = command_colors.get(command)
                if color is None and command.startswith("COLOR CHANGE"):
                    color = command_colors["COLOR CHANGE"]
                if color is None:
                    color = wx.Colour(80, 80, 80)
                dc.SetBrush(wx.Brush(color))
                dc.SetPen(wx.Pen(color, 1))
                dc.DrawPolygon(marker)

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
        commands = self.viewer.command_events.get(vis, ())
        if commands:
            txt_left += f" | {' | '.join(commands)}"
        txt_center = f"{vis / total * 100:.1f}%"
        if hasattr(self.viewer, "bounds") and self.viewer.bounds != (0, 0, 0, 0):
            bw = self.viewer.bounds[2] - self.viewer.bounds[0]
            bh = self.viewer.bounds[3] - self.viewer.bounds[1]
            txt_right = (
                f"{bw:.1f} x {bh:.1f} mm | "
                f"{self.viewer.color_count} color sections"
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
        self.show_stitches = True
        self.show_realistic = False
        self.show_density = False
        self.show_jumps = False
        self.risky_jumps_only = False
        self.show_needle = True
        self.needle_highlighted = False
        self.needle_highlight_stage = 0
        self._needle_highlight_timer = None
        self.stitches_np = np.zeros((0, 7), dtype=np.float32)
        self.bounds = (0, 0, 0, 0)
        self.color_boundaries = []
        self.color_count = 0
        self.command_events = {}
        self.jump_segments = []
        self.stitch_points_np = np.zeros((0, 2), dtype=np.float32)
        self.stitch_density_np = np.zeros((0,), dtype=np.float32)
        self.density_ready = False
        self.cached_bitmap = None
        self.cached_pan_x = self.pan_x
        self.cached_pan_y = self.pan_y
        self.cached_zoom = self.zoom
        self.zoom_render_timer = None
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

    def SetOneToOne(self):
        """Display the design at its physical size when display PPI is known."""
        if self.stitches_np.shape[0] == 0:
            return
        try:
            display_index = wx.Display.GetFromWindow(self)
            if display_index == wx.NOT_FOUND:
                display_index = 0
            ppi = wx.Display(display_index).GetPPI()
            ppi_x = float(ppi.x)
            ppi_y = float(ppi.y)
            if ppi_x <= 0 or ppi_y <= 0:
                raise ValueError("invalid display PPI")
            pixels_per_mm = (ppi_x + ppi_y) / (2.0 * 25.4)
        except (AttributeError, TypeError, ValueError, wx.PyNoAppError):
            pixels_per_mm = 96.0 / 25.4
        self.zoom = pixels_per_mm
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

    def JumpToCommand(self, direction):
        """Move to the nearest recorded JUMP, TRIM, or color-change event."""
        positions = sorted(self.command_events)
        current = self.visible_count
        if direction > 0:
            targets = (position for position in positions if position > current)
        else:
            targets = (
                position for position in reversed(positions) if position < current
            )
        target = next(targets, None)
        if target is None:
            target = self.stitches_np.shape[0] if direction > 0 else 0
            if target == current:
                return False
        self.visible_count = target
        return True

    def RotateDesign(self, quarter_turns):
        """Rotate the loaded design by quarter turns around its center."""
        if self.stitches_np.shape[0] == 0:
            return

        min_x, min_y, max_x, max_y = self.bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        turns = quarter_turns % 4
        if turns == 0:
            return

        def rotate_coordinates(coordinates):
            relative_x = coordinates[:, 0] - center_x
            relative_y = coordinates[:, 1] - center_y
            if turns == 1:
                coordinates[:, 0] = center_x - relative_y
                coordinates[:, 1] = center_y + relative_x
            elif turns == 2:
                coordinates[:, 0] = center_x - relative_x
                coordinates[:, 1] = center_y - relative_y
            else:
                coordinates[:, 0] = center_x + relative_y
                coordinates[:, 1] = center_y - relative_x

        rotate_coordinates(self.stitches_np[:, 0:2])
        rotate_coordinates(self.stitches_np[:, 2:4])
        rotate_coordinates(self.stitch_points_np)
        if self.jump_segments:
            jump_coordinates = np.asarray(self.jump_segments, dtype=np.float32)
            rotate_coordinates(jump_coordinates[:, 0:2])
            rotate_coordinates(jump_coordinates[:, 2:4])
            self.jump_segments = jump_coordinates.tolist()

        rotated_corners = np.array(
            [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]],
            dtype=np.float32,
        )
        rotate_coordinates(rotated_corners)
        self.bounds = (
            float(rotated_corners[:, 0].min()),
            float(rotated_corners[:, 1].min()),
            float(rotated_corners[:, 0].max()),
            float(rotated_corners[:, 1].max()),
        )
        self.need_redraw = True
        self.CenterDesign()

    def OnKeyDown(self, e):
        """Handle playback, navigation, display, and view shortcut keys."""
        now = time.time()
        key = e.GetKeyCode()
        is_alt = e.AltDown()
        is_ctrl = e.ControlDown()
        # Let menu mnemonics and global shortcuts pass through
        # Alt+F, Alt+P for menu, Ctrl+Q for Quit, Ctrl+O for Open etc.
        if is_alt and key in (ord('F'), ord('f'), ord('P'), ord('p')):
            e.Skip()
            return
        if is_ctrl and key in (ord('Q'), ord('q'), ord('O'), ord('o')):
            e.Skip()
            return
        is_space_or_c = key in (
            wx.WXK_SPACE,
            ord("C"),
            ord("c"),
        )
        if (
            not is_space_or_c
            and now - self._last_key_time < self._key_throttle
            and not is_alt
            and not is_ctrl
        ):
            return
        self._last_key_time = now
        total = self.stitches_np.shape[0]
        is_shift = e.ShiftDown()
        changed = False
        highlight_needle = False
        step = 1 if is_alt else self.step_size
        if is_shift and not is_alt and not is_ctrl and key in (
            wx.WXK_RIGHT,
            wx.WXK_LEFT,
        ):
            changed = self.JumpToCommand(
                1 if key == wx.WXK_RIGHT else -1
            )
            highlight_needle = changed
            if changed and self.is_playing:
                self.play_timer.Stop()
                self.is_playing = False
        elif self.is_playing and not is_alt and not is_ctrl and key in (
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
            highlight_needle = True
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
        elif key in (ord("C"), ord("c")) and not is_alt and not is_ctrl:
            self.CenterDesign()
            return
        elif key in (ord("F"), ord("f")) and not is_alt and not is_ctrl:
            self.FitToScreen()
            return
        elif key == ord("1") and not is_alt and not is_ctrl:
            self.SetOneToOne()
            return
        elif key == wx.WXK_F11:
            frame = wx.GetTopLevelParent(self)
            if hasattr(frame, "ToggleFullScreen"):
                frame.ToggleFullScreen()
                return
        elif key in (ord("G"), ord("g")) and not is_alt and not is_ctrl:
            self.show_grid = not self.show_grid
            frame = wx.GetTopLevelParent(self)
            if hasattr(frame, "gridItem"):
                frame.gridItem.Check(self.show_grid)
            changed = True
        elif key in (ord("J"), ord("j")) and not is_alt and not is_ctrl:
            if not self.show_jumps:
                self.show_jumps = True
                self.risky_jumps_only = False
            elif not self.risky_jumps_only:
                self.risky_jumps_only = True
            else:
                self.show_jumps = False
                self.risky_jumps_only = False
            changed = True
        elif key in (ord("X"), ord("x")) and not is_alt and not is_ctrl:
            self.show_density = not self.show_density
            if self.show_density and not self.density_ready:
                self.CalculateStitchDensity()
            changed = True
        elif key in (ord("V"), ord("v")) and not is_alt and not is_ctrl:
            self.show_stitches = not self.show_stitches
            changed = True
        elif key in (ord("R"), ord("r")) and not is_alt and not is_ctrl:
            self.show_realistic = not self.show_realistic
            frame = wx.GetTopLevelParent(self)
            if hasattr(frame, "realisticItem"):
                frame.realisticItem.Check(self.show_realistic)
            changed = True
        elif key in (ord("N"), ord("n")) and not is_alt and not is_ctrl:
            self.show_needle = not self.show_needle
            if self.show_needle:
                self.HighlightNeedle()
            else:
                self.StopNeedleHighlight()
            changed = True
        elif key in (ord("H"), ord("h")) and not is_alt and not is_ctrl:
            self.ShowHelp()
            return
        elif key in (ord("I"), ord("i")) and not is_alt and not is_ctrl:
            self.ShowSettings()
            return
        elif key == wx.WXK_ESCAPE:
            if self.is_playing:
                self.play_timer.Stop()
                self.is_playing = False
                return
        if changed:
            if highlight_needle:
                self.HighlightNeedle()
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
        help_header = f"{APP_TITLE}\n{'=' * len(APP_TITLE)}\n\n"
        help_text = help_header + (
            "MOUSE\n"
            "  Wheel       Zoom\n"
            "  Drag        Pan\n"
            "  Click bar   Seek\n\n"
            "PLAYBACK\n"
            "  Right/Left       Speed up/down while playing\n"
            "  Alt+Right/Left   Move by one stitch when stopped\n"
            "  Shift+Right/Left Next/previous command\n"
            "  Ctrl+Right/Left  Next/previous color\n"
            "  Up/Down          Fast seek when stopped\n"
            "  Home/End         First/last stitch\n"
            "  Space            Play/pause\n"
            "  Esc              Stop\n\n"
            "VIEW\n"
            "  C                Center design\n"
            "  F                Fit design to window\n"
            "  F11              Toggle fullscreen\n"
            "  1                Show design at physical 1:1 size\n"
            "  V                Toggle embroidery visibility\n"
            "  G                Toggle grid\n"
            "  H                Show this help\n"
            "  I                Show viewer settings\n\n"
            "  N                Toggle needle crosshair\n\n"
            "  J                Toggle jump paths\n"
            "  X                Toggle stitch density map\n\n"
            "RENDERING\n"
            "  +/-              Change thread width\n"
            "  [/]              Change dark shading\n"
            "  Shift+[/]        Change light shading\n"
        )
        dlg = wx.MessageDialog(self, help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def ShowSettings(self):
        """Show the current viewer state and rendering parameters."""
        total = self.stitches_np.shape[0]
        min_x, min_y, max_x, max_y = self.bounds
        width = max_x - min_x
        height = max_y - min_y
        density_mode = "on" if self.show_density else "off"
        jump_mode = (
            "risky only" if self.risky_jumps_only
            else "all" if self.show_jumps
            else "off"
        )
        settings_text = (
            f"{APP_TITLE} viewer settings\n\n"
            f"Design\n"
            f"  Stitches: {self.visible_count}/{total}\n"
            f"  Color sections: {self.color_count}\n"
            f"  Bounds: {width:.1f} x {height:.1f} mm\n\n"
            f"Viewport\n"
            f"  Zoom: {self.zoom:.3f}\n"
            f"  Pan: {self.pan_x:.1f}, {self.pan_y:.1f} px\n"
            f"  Grid: {'on' if self.show_grid else 'off'}\n"
            f"  Embroidery: {'on' if self.show_stitches else 'off'}\n"
            f"  Realistic render: {'on' if self.show_realistic else 'off'}\n"
            f"  Jump paths: {jump_mode}\n"
            f"  Density map: {density_mode}\n"
            f"  Density radius: {DENSITY_RADIUS_MM:.1f} mm\n"
            f"  Density warning: {DENSITY_WARNING_PER_MM2:.1f} stitches/mm2\n"
            f"  Density critical: {DENSITY_CRITICAL_PER_MM2:.1f} stitches/mm2\n"
            f"  Needle: {'on' if self.show_needle else 'off'}\n"
            f"  Gradient: {'on' if self.zoom > 1.2 else 'off'}\n\n"
            f"Rendering\n"
            f"  Line width: {self.line_width:.1f} mm\n"
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
        has_thread_colors = bool(
            hasattr(pattern, "threadlist") and pattern.threadlist
        )
        palette = pattern.threadlist if has_thread_colors else AUTO_THREAD_COLORS
        min_x = min_y = 1e9
        max_x = max_y = -1e9
        self.color_boundaries = [0]
        self.color_count = 0
        self.command_events = {}
        self.jump_segments = []
        self.stitch_points_np = np.zeros((0, 2), dtype=np.float32)
        self.stitch_density_np = np.zeros((0,), dtype=np.float32)
        self.density_ready = False
        jump_run_indices = []
        for st in pattern.stitches:
            x = st[0] / 10.0
            y = st[1] / 10.0
            raw_command = st[2] if len(st) > 2 else emb.STITCH
            if hasattr(emb, "decode_embroidery_command"):
                cmd, thread, needle, order = emb.decode_embroidery_command(
                    raw_command
                )
            else:
                cmd = raw_command
                thread = needle = order = None
            if hasattr(emb, "JUMP") and cmd == emb.JUMP:
                event_position = len(segs)
                self.command_events.setdefault(event_position, []).append("JUMP")
                self.jump_segments.append([last_x, last_y, x, y, 1, len(segs)])
                jump_run_indices.append(len(self.jump_segments) - 1)
                last_x, last_y = x, y
                continue
            if hasattr(emb, "END") and cmd == emb.END:
                break
            is_color_change = (
                hasattr(emb, "COLOR_CHANGE") and cmd == emb.COLOR_CHANGE
            )
            if is_color_change:
                for jump_index in jump_run_indices:
                    self.jump_segments[jump_index][4] = 0
                jump_run_indices = []
                event_position = len(segs)
                details = []
                if thread is not None:
                    details.append(f"T{thread}")
                if needle is not None:
                    details.append(f"N{needle}")
                if order is not None:
                    details.append(f"O{order}")
                command_label = "COLOR CHANGE"
                if details:
                    command_label += f" ({', '.join(details)})"
                self.command_events.setdefault(event_position, []).append(
                    command_label
                )
                cur_color_idx += 1
                if segs:
                    self.color_boundaries.append(len(segs))
                last_x, last_y = x, y
                continue
            if hasattr(emb, "TRIM") and cmd == emb.TRIM:
                event_position = len(segs)
                self.command_events.setdefault(event_position, []).append("TRIM")
                continue
            for command_name in ("STOP", "SLOW", "FAST"):
                if hasattr(emb, command_name) and cmd == getattr(emb, command_name):
                    event_position = len(segs)
                    self.command_events.setdefault(event_position, []).append(
                        command_name
                    )
                    break
            else:
                command_name = None
            if command_name is not None:
                continue
            if has_thread_colors:
                color_idx = min(cur_color_idx, len(palette) - 1)
            else:
                color_idx = cur_color_idx % len(AUTO_THREAD_COLORS)
            col = palette[color_idx]
            if hasattr(col, "get_red"):
                rgb = (col.get_red(), col.get_green(), col.get_blue())
            elif isinstance(col, (list, tuple)):
                rgb = tuple(col[:3])
            else:
                rgb = AUTO_THREAD_COLORS[color_idx % len(AUTO_THREAD_COLORS)]
            segs.append((last_x, last_y, x, y, rgb[0], rgb[1], rgb[2]))
            min_x = min(min_x, last_x, x)
            min_y = min(min_y, last_y, y)
            max_x = max(max_x, last_x, x)
            max_y = max(max_y, last_y, y)
            last_x, last_y = x, y
        if segs:
            self.stitches_np = np.array(segs, dtype=np.float32)
            self.stitch_points_np = self.stitches_np[:, 2:4].copy()
            self.bounds = (min_x, min_y, max_x, max_y)
            self.visible_count = self.stitches_np.shape[0]
            self.color_boundaries = sorted(
                set(boundary for boundary in self.color_boundaries if boundary < len(segs))
            )
            self.color_count = len(self.color_boundaries)
            if fit_to_screen:
                self._pending_fit_to_screen = True
                wx.CallAfter(self._try_fit_to_screen)
        self.need_redraw = True
        self.Refresh()
        if self.progress_bar:
            self.progress_bar.Refresh()
        self.SetFocus()
        return True

    def CalculateStitchDensity(self):
        """Calculate the density map once, on demand, using the Numba kernel."""
        if self.density_ready or len(self.stitch_points_np) == 0:
            return
        frame = wx.GetTopLevelParent(self)
        if hasattr(frame, "SetStatusText"):
            frame.SetStatusText("Calculating stitch density...")
        wx.BeginBusyCursor()
        wx.SafeYield(frame, True)
        min_x, min_y, max_x, max_y = self.bounds
        try:
            self.stitch_density_np = calculate_stitch_density_numba(
                self.stitch_points_np,
                min_x,
                min_y,
                max_x,
                max_y,
            )
        finally:
            wx.EndBusyCursor()
        self.density_ready = True
        if hasattr(frame, "SetStatusText"):
            frame.SetStatusText("Density map ready")
            wx.CallLater(1500, frame.SetStatusText, DEFAULT_STATUS_TEXT)
        self.need_redraw = True
        self.Refresh()

    def OnPaint(self, e):
        """Render the current viewport, using the cached bitmap when possible."""
        dc = wx.PaintDC(self)
        dc.Clear()
        if self._pending_fit_to_screen:
            return
        if not self.need_redraw and self.cached_bitmap:
            zoom_ratio = self.zoom / self.cached_zoom
            if abs(zoom_ratio - 1.0) < 0.001:
                pan_delta_x = round(self.pan_x - self.cached_pan_x)
                pan_delta_y = round(self.pan_y - self.cached_pan_y)
                dc.DrawBitmap(self.cached_bitmap, pan_delta_x, pan_delta_y)
            else:
                bitmap_width = self.cached_bitmap.GetWidth()
                bitmap_height = self.cached_bitmap.GetHeight()
                preview_x = round(self.pan_x - zoom_ratio * self.cached_pan_x)
                preview_y = round(self.pan_y - zoom_ratio * self.cached_pan_y)
                source_dc = wx.MemoryDC()
                source_dc.SelectObject(self.cached_bitmap)
                try:
                    dc.StretchBlit(
                        preview_x,
                        preview_y,
                        round(bitmap_width * zoom_ratio),
                        round(bitmap_height * zoom_ratio),
                        source_dc,
                        0,
                        0,
                        bitmap_width,
                        bitmap_height,
                    )
                finally:
                    source_dc.SelectObject(wx.NullBitmap)
            self.DrawAnalysisOverlays(dc)
            self.DrawNeedleOverlay(dc)
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
        use_shaded = self.zoom > 1.2
        buf = np.full((h, w, 3), 255, dtype=np.uint8)
        if self.show_grid:
            render_grid_numba(buf, self.zoom, self.pan_x, self.pan_y)
        if self.show_stitches and self.stitches_np.shape[0] > 0 and self.visible_count > 0:
            render_shaded_numba(
                buf,
                self.stitches_np,
                self.visible_count,
                self.zoom,
                self.pan_x,
                self.pan_y,
                use_shaded,
                self.line_width,
                self.dark_factor,
                self.light_factor,
                self.show_realistic and self.zoom > 1.2,
            )
        if self.show_density and len(self.stitch_points_np) > 0:
            render_density_numba(
                buf,
                self.stitch_points_np,
                self.stitch_density_np,
                self.visible_count,
                self.zoom,
                self.pan_x,
                self.pan_y,
            )
        img = wx.Image(w, h)
        img.SetData(buf.tobytes())
        bmp = wx.Bitmap(img)
        self.cached_bitmap = bmp
        self.cached_pan_x = self.pan_x
        self.cached_pan_y = self.pan_y
        self.cached_zoom = self.zoom
        self.need_redraw = False
        dc.DrawBitmap(bmp, 0, 0)
        self.DrawAnalysisOverlays(dc)
        self.DrawNeedleOverlay(dc)

    def DrawAnalysisOverlays(self, dc):
        """Draw optional jump paths and local stitch-density diagnostics."""
        if self.show_jumps:
            for x1, y1, x2, y2, risky, stitch_index in self.jump_segments:
                if stitch_index > self.visible_count:
                    continue
                if self.risky_jumps_only and not risky:
                    continue
                color = wx.Colour(220, 45, 45) if risky else wx.Colour(100, 100, 100)
                dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_SHORT_DASH))
                dc.DrawLine(
                    int(x1 * self.zoom + self.pan_x),
                    int(y1 * self.zoom + self.pan_y),
                    int(x2 * self.zoom + self.pan_x),
                    int(y2 * self.zoom + self.pan_y),
                )


    def DrawNeedleOverlay(self, dc):
        """Draw the current needle position above the cached stitch bitmap."""
        if not self.show_stitches or not self.show_needle or self.stitches_np.shape[0] == 0:
            return
        if self.visible_count > 0:
            stitch = self.stitches_np[self.visible_count - 1]
            world_x, world_y = stitch[2], stitch[3]
        else:
            stitch = self.stitches_np[0]
            world_x, world_y = stitch[0], stitch[1]
        needle_x = int(world_x * self.zoom + self.pan_x)
        needle_y = int(world_y * self.zoom + self.pan_y)
        if self.needle_highlight_stage == 2:
            arm, radius, outer_radius = 80, 24, 42
        elif self.needle_highlight_stage == 1:
            arm, radius, outer_radius = 48, 16, 28
        else:
            arm, radius, outer_radius = 14, 6, 0
        dc.SetPen(wx.Pen(wx.Colour(10, 10, 10), 8 if outer_radius else 4))
        dc.DrawLine(needle_x - arm, needle_y, needle_x + arm, needle_y)
        dc.DrawLine(needle_x, needle_y - arm, needle_x, needle_y + arm)
        if outer_radius:
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.DrawCircle(needle_x, needle_y, outer_radius)
        dc.SetPen(wx.Pen(wx.Colour(255, 255, 255), 3 if outer_radius else 2))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawCircle(needle_x, needle_y, radius)
        dc.DrawLine(needle_x - arm, needle_y, needle_x + arm, needle_y)
        dc.DrawLine(needle_x, needle_y - arm, needle_x, needle_y + arm)
        dc.SetBrush(wx.Brush(wx.Colour(255, 220, 40)))
        dc.SetPen(wx.Pen(wx.Colour(10, 10, 10), 2))
        dc.DrawCircle(needle_x, needle_y, 5 if outer_radius else 3)

    def HighlightNeedle(self):
        """Pulse a large needle marker after user navigation."""
        if not self.show_needle:
            return
        self.needle_highlighted = True
        self.needle_highlight_stage = 2
        if self._needle_highlight_timer is not None:
            self._needle_highlight_timer.Stop()
        self._needle_highlight_timer = wx.CallLater(
            200,
            self._SetNeedleHighlightStage,
            1,
        )
        self.Refresh()

    def _SetNeedleHighlightStage(self, stage):
        """Advance the temporary needle marker through its visual pulse."""
        if not self.show_needle:
            return
        self.needle_highlight_stage = stage
        self.Refresh()
        if stage == 1:
            self._needle_highlight_timer = wx.CallLater(
                300,
                self.StopNeedleHighlight,
            )

    def StopNeedleHighlight(self):
        """Return the needle crosshair to its normal size."""
        self.needle_highlighted = False
        self.needle_highlight_stage = 0
        self._needle_highlight_timer = None
        self.Refresh()

    def OnWheel(self, e):
        """Zoom around the mouse position while preserving its world point."""
        mx, my = e.GetPosition()
        old = self.zoom
        self.zoom *= 1.15 if e.GetWheelRotation() > 0 else 1 / 1.15
        self.zoom = max(0.05, min(50.0, self.zoom))
        scale = self.zoom / old
        self.pan_x = mx - scale * (mx - self.pan_x)
        self.pan_y = my - scale * (my - self.pan_y)
        if self.zoom_render_timer is not None:
            self.zoom_render_timer.Stop()
        if self.cached_bitmap:
            self.need_redraw = False
            self.zoom_render_timer = wx.CallLater(
                140,
                self._finish_zoom_render,
            )
        else:
            self.need_redraw = True
        self.Refresh()

    def _finish_zoom_render(self):
        """Schedule a full-quality render after zooming settles."""
        self.zoom_render_timer = None
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
        self.need_redraw = True
        self.Refresh()
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
        batch=False,
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
        # TODO: Consider migrating wx.Config to an explicit XDG config path.
        self.config = wx.Config(APP_TITLE)
        self.last_directory = self.config.Read("last_directory", "")
        self.current_file_path = None
        if initial_file and Path(initial_file).is_file():
            self.current_file_path = Path(initial_file).resolve()
            self.last_directory = str(self.current_file_path.parent)
            self.config.Write("last_directory", self.last_directory)
            self.config.Flush()

        # Create the main panel, viewer, and progress bar, and arrange them vertically.
        main_panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.viewer = EmbroideryViewerPanel(main_panel, None)
        self.progress = ProgressBarPanel(main_panel, self.viewer)
        self.viewer.progress_bar = self.progress
        self.viewer.SetDropTarget(EmbroideryFileDropTarget(self))

        sizer.Add(self.viewer, 1, wx.EXPAND)
        sizer.Add(self.progress, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
                  6)
        main_panel.SetSizer(sizer)
        self._main_panel = main_panel

        # Build the menu bar with file and playback options, and bind them to handlers.
        menubar = wx.MenuBar()

        fileMenu = wx.Menu()
        openItem = fileMenu.Append(wx.ID_OPEN, "Open embroidery file\tCtrl+O")
        exportPrintItem = fileMenu.Append(wx.ID_ANY, "Export PNG for print...")
        exportShadedItem = fileMenu.Append(
            wx.ID_ANY, "Export shaded PNG for print..."
        )
        exportIconItem = fileMenu.Append(wx.ID_ANY, "Export preview PNG...")
        centerItem = fileMenu.Append(wx.ID_ANY, "Center design\tC")
        fitItem = fileMenu.Append(wx.ID_ANY, "Fit design to window\tF")
        fullscreenItem = fileMenu.Append(wx.ID_ANY, "Fullscreen\tF11")
        gridItem = fileMenu.AppendCheckItem(wx.ID_ANY, "Show 1cm grid\tG")
        gridItem.Check(True)
        realisticItem = fileMenu.AppendCheckItem(
            wx.ID_ANY, "Realistic thread render\tR"
        )
        helpItem = fileMenu.Append(wx.ID_ANY, "Help\tH")
        fileMenu.AppendSeparator()
        rotateLeftItem = fileMenu.Append(wx.ID_ANY, "Rotate left 90 deg")
        rotateRightItem = fileMenu.Append(wx.ID_ANY, "Rotate right 90 deg")
        fileMenu.AppendSeparator()
        quitItem = fileMenu.Append(wx.ID_EXIT, "Quit\tCtrl+Q")
        menubar.Append(fileMenu, "&File")

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
        menubar.Append(playbackMenu, "&Playback")
        self.SetMenuBar(menubar)

        # Store menubar reference for key handling
        self.menubar = menubar
        self.fileMenu = fileMenu
        self.playbackMenu = playbackMenu
        self.gridItem = gridItem
        self.realisticItem = realisticItem

        # Global accelerators
        # Alt+F / Alt+P are handled by mnemonics in menu titles (&File / &Playback).
        # wxWidgets automatically exposes them as Alt+F and Alt+P.
        # Ctrl+Q for Quit is added explicitly via AcceleratorTable.
        accel_tbl = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('Q'), quitItem.GetId()),
        ])
        self.SetAcceleratorTable(accel_tbl)

        # Ensure Ctrl+Q works even when viewer has focus.
        # Alt+F / Alt+P are left to native menu bar mnemonics (no PopupMenu on attached menu).
        self.Bind(wx.EVT_CHAR_HOOK, self.OnCharHook)

        # Bind menu items to their handlers.
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_MENU, self.OnOpen, openItem)
        self.Bind(wx.EVT_MENU, self.ExportPrintPng, exportPrintItem)
        self.Bind(wx.EVT_MENU, self.ExportShadedPng, exportShadedItem)
        self.Bind(wx.EVT_MENU, self.ExportIconPng, exportIconItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.CenterDesign(), centerItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.FitToScreen(), fitItem)
        self.Bind(wx.EVT_MENU, lambda e: self.ToggleFullScreen(), fullscreenItem)
        self.Bind(wx.EVT_MENU, self.OnToggleGrid, gridItem)
        self.Bind(wx.EVT_MENU, self.OnToggleRealistic, realisticItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.ShowHelp(), helpItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.RotateDesign(-1), rotateLeftItem)
        self.Bind(wx.EVT_MENU, lambda e: self.viewer.RotateDesign(1), rotateRightItem)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), quitItem)
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
        self.SetStatusText(DEFAULT_STATUS_TEXT)

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
            and Path(initial_file).exists()
            and self.viewer.LoadDesign(initial_file, fit_to_screen=False)
        )
        if initial_file_loaded:
            total = self.viewer.stitches_np.shape[0]
            self.SetTitle(
                f"{APP_TITLE} - {Path(initial_file).name} - {total} sts"
            )

        if batch:
            return
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

    def OnCharHook(self, e):
        """Global keyboard shortcuts for menu.

        - Ctrl+Q -> Quit
        - Alt+F / Alt+P are handled natively by menubar mnemonics (&File, &Playback)
          so we just skip them here to let wxWidgets process them.
        """
        kc = e.GetKeyCode()
        # Ctrl+Q
        if e.ControlDown() and kc in (ord('Q'), ord('q')):
            self.Close()
            return
        # For Alt+F and Alt+P, do not intercept with PopupMenu (causes
        # !IsAttached() assertion on attached menus). Let the native
        # menubar mnemonic handling do its job.
        if e.AltDown() and kc in (ord('F'), ord('f'), ord('P'), ord('p')):
            e.Skip()
            return
        e.Skip()

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

    def OnToggleRealistic(self, e):
        """Toggle the 2.5D realistic thread renderer."""
        self.viewer.show_realistic = e.IsChecked()
        self.viewer.need_redraw = True
        self.viewer.Refresh()

    def OnOpen(self, e):
        """Prompt for an embroidery file and update the window metadata."""
        dlg = EmbroideryOpenDialog(
            self,
            self.last_directory,
            self.current_file_path,
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.OpenFile(dlg.GetPath())
        dlg.Destroy()

    def _choose_export_path(self, title):
        """Ask for a PNG destination and return it, or None if cancelled."""
        dlg = wx.FileDialog(
            self,
            title,
            wildcard="PNG files (*.png)|*.png",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return None
            path = Path(dlg.GetPath())
        finally:
            dlg.Destroy()
        return path.with_suffix(".png")

    def ExportPng(self, path, icon=False, dpi=300, background="transparent",
                  grid=False, shaded=False):
        """Export clean embroidery geometry as a PNG file."""
        if self.viewer.stitches_np.shape[0] == 0:
            return False
        if icon:
            width = height = 256
        else:
            min_x, min_y, max_x, max_y = self.viewer.bounds
            width = max(1, round((max_x - min_x) / 25.4 * dpi))
            height = max(1, round((max_y - min_y) / 25.4 * dpi))
        image, metadata = render_export_image(
            self.viewer.stitches_np,
            self.viewer.bounds,
            width,
            height,
            self.viewer.line_width,
            dpi=dpi,
            background=background,
            grid=grid,
            shaded=shaded,
            dark_factor=self.viewer.dark_factor,
            light_factor=self.viewer.light_factor,
        )
        image.save(path, "PNG", pnginfo=metadata, dpi=(dpi, dpi))
        return True

    def ExportPrintPng(self, e):
        """Export a flat 300 DPI PNG at the design's physical size."""
        path = self._choose_export_path("Export PNG for print")
        if path:
            self.ExportPng(path, dpi=300)

    def ExportShadedPng(self, e):
        """Export a shaded 300 DPI PNG at the design's physical size."""
        path = self._choose_export_path("Export shaded PNG for print")
        if path:
            self.ExportPng(path, dpi=300, shaded=True)

    def ExportIconPng(self, e):
        """Export a 256 pixel transparent preview PNG."""
        path = self._choose_export_path("Export preview PNG")
        if path:
            self.ExportPng(path, icon=True, dpi=96)

    def OpenFile(self, path):
        """Load a file and update window metadata after a successful load."""
        selected_path = Path(path).resolve()
        if not self.viewer.LoadDesign(str(selected_path), fit_to_screen=True):
            return False
        self.current_file_path = selected_path
        self.last_directory = str(selected_path.parent)
        self.config.Write("last_directory", self.last_directory)
        self.config.Flush()
        total = self.viewer.stitches_np.shape[0]
        bw = self.viewer.bounds[2] - self.viewer.bounds[0]
        bh = self.viewer.bounds[3] - self.viewer.bounds[1]
        self.SetTitle(
            f"{APP_TITLE} - {selected_path.name} - {total} sts - {bw:.1f}x{bh:.1f}mm"
        )
        self.progress.Refresh()
        return True

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
    parser.add_argument(
        "--export-png",
        metavar="PATH",
        help="Export a clean print PNG and exit",
    )
    parser.add_argument(
        "--export-shaded-png", "--png",
        metavar="PATH",
        help="Export a shaded print PNG and exit",
    )
    parser.add_argument(
        "--export-icon", "--icon",
        metavar="PATH",
        help="Export a clean 256px preview PNG and exit",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for --export-png (default: 300)",
    )
    parser.add_argument(
        "--export-background", "--bg",
        choices=("transparent", "white"),
        default="transparent",
        help="PNG background (default: transparent)",
    )
    parser.add_argument(
        "--export-grid", "--grid",
        action="store_true",
        help="Add a 10 mm grid to exported PNG",
    )
    args=parser.parse_args()

    export_paths = [
        path for path in (
            args.export_png,
            args.export_shaded_png,
            args.export_icon,
        )
        if path
    ]
    if len(export_paths) > 1:
        parser.error("choose only one export option at a time")
    if args.dpi <= 0:
        parser.error("DPI must be greater than zero")
    export_requested = bool(export_paths)
    if export_requested and not args.input_file:
        parser.error(
            "an input embroidery file is required for export; "
            "use: inksim INPUT_FILE --export-png OUTPUT.png"
        )
    if args.input_file and not Path(args.input_file).is_file():
        parser.error(f"input embroidery file not found: {args.input_file}")

    window_size = args.size
    window_position = args.position
    app=wx.App(not export_requested)
    frame = Frame(
        initial_file=args.input_file,
        fullscreen=args.fullscreen,
        window_size=window_size,
        window_position=window_position,
        autoplay=args.play,
        batch=export_requested,
    )
    if export_requested:
        export_path = export_paths[0]
        success = frame.ExportPng(
            export_path,
            icon=bool(args.export_icon),
            dpi=96 if args.export_icon else args.dpi,
            background=args.export_background,
            grid=args.export_grid,
            shaded=bool(args.export_shaded_png),
        )
        frame.Destroy()
        raise SystemExit(0 if success else 1)
    app.MainLoop()
