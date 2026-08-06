# InkSim

InkSim is a standalone interactive embroidery simulator and preview renderer.
It opens embroidery files, displays their stitch sequence, and lets the user
inspect or replay the design before production. It is implemented as a small
Python/wxPython/Numba application and is independent of the main Ink/Stitch GUI.

The program is useful for:

- quickly checking stitch order, thread colors, jumps, trims, and commands;
- replaying a design stitch by stitch or at adjustable playback speeds;
- inspecting the design at fit-to-window, physical 1:1, or custom zoom levels;
- previewing the design on a procedural fabric background;
- exporting clean PNG previews for print or documentation.

## Requirements

- system Python3 or any virtual environment;
- wxPython;
- NumPy and Numba;
- Pillow;
- pystitch.

The repository already contains the project dependency configuration. From the
repository root, use the project environment:

```bash
bin/inksim.py
```

The script also contains a fallback to virtual-environment bootstrap. When it is
started with a system Python and a project `.venv` exists, it re-executes
itself with that interpreter.

## Basic Usage

Open an empty viewer:

```bash
bin/inksim.py
```

Open a design immediately:

```bash
bin/inksim.py design.dst
```

Start playback from the first stitch:

```bash
bin/inksim.py design.dst --play
```

Start fullscreen or choose an explicit window geometry:

```bash
bin/inksim.py design.dst --fullscreen
bin/inksim.py design.dst --size 1600x1000 --position 100,50
```

## Interactive Controls

### Mouse and Window

| Action                     | Function                  |
| -------------------------- | ------------------------- |
| Mouse wheel                | Zoom around the cursor    |
| Left-drag in viewer        | Pan the design            |
| Click or drag the timeline | Seek to a stitch position |
| Drop a file on the viewer  | Open the file             |
| `F11`                      | Toggle fullscreen         |

### Playback and Navigation

| Shortcut                     | Function                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `Space`                      | Play or pause                                                                 |
| `Right` / `Left`             | Move by the configured step when stopped; change playback speed while playing |
| `Alt+Right` / `Alt+Left`     | Move one stitch                                                               |
| `Up` / `Down`                | Move by ten configured steps                                                  |
| `Home` / `End`               | Move to the first or last stitch                                              |
| `Ctrl+Right` / `Ctrl+Left`   | Move to the next or previous color section                                    |
| `Shift+Right` / `Shift+Left` | Move to the next or previous command event                                    |
| `Esc`                        | Stop playback                                                                 |

The playback step can also be selected from the Playback menu: 1, 10, 50,
100, or 500 stitches.

### View and Analysis

| Shortcut              | Function                                                   |
| --------------------- | ---------------------------------------------------------- |
| `C`                   | Center the design                                          |
| `F`                   | Fit the design to the viewer                               |
| `1`                   | Display at physical 1:1 size when display PPI is available |
| `G`                   | Toggle the 1 cm helper grid                                |
| `V`                   | Toggle embroidery visibility                               |
| `R`                   | Toggle realistic thread rendering and fabric background    |
| `J`                   | Cycle jumps: off, all jumps, risky jumps only              |
| `X`                   | Toggle the stitch-density map                              |
| `N`                   | Toggle the needle marker                                   |
| `H`                   | Show help                                                  |
| `I`                   | Show current viewer settings                               |
| `+` / `-`             | Increase or decrease thread width                          |
| `[` / `]`             | Adjust dark shading                                        |
| `Shift+[` / `Shift+]` | Adjust light shading                                       |

## Rendering Architecture

InkSim keeps the loaded design in a NumPy array with one row per stitch
segment:

```text
[x1, y1, x2, y2, red, green, blue]
```

Coordinates are converted from pystitch units to millimeters during loading.
The viewer then projects millimeters to screen pixels using:

```text
screen_x = world_x * zoom + pan_x
screen_y = world_y * zoom + pan_y
```

The image is rendered into an RGB NumPy buffer and converted to a wxPython
bitmap. Numba kernels perform the pixel-heavy work.

### Flat and Shaded Rendering

`render_shaded_numba` is the normal fast path. It supports both flat colors
and a lightweight longitudinal gradient. It rasterizes each stitch segment
with a bounded line width and a small anti-aliased edge. The renderer is kept
deliberately simple so it remains responsive during playback and navigation.

### Realistic Rendering

When `R` is enabled and the zoom is high enough for detail to be meaningful,
InkSim renders a procedural fabric background and routes the stitch data to
`render_realistic_numba`. The realistic path is separate from the normal
renderer and includes:

- a cylindrical cross-section for each thread;
- diffuse lighting from a fixed top-left light direction;
- a specular highlight for thread sheen;
- a small longitudinal twist modulation;
- soft cast shadows on the fabric;
- anti-aliased thread edges;
- zoom-aware fabric relief and low-zoom texture suppression to reduce moire.

The realistic renderer is an intentionally approximate per-stitch model. Its
isolated cylinders can exaggerate sewing direction and dark gaps, especially
in satin areas. A future photorealistic implementation should treat a satin
column as one continuous anisotropic surface or use a normal map instead of
shading every microscopic stitch as an independent cylinder.

### Analysis Overlays

Analysis overlays are drawn after the cached bitmap:

- jump paths are shown as dashed lines;
- risky jumps are distinguished from jumps associated with color changes;
- the density map colors stitch endpoints by local stitch density;
- the needle marker shows the current endpoint and briefly enlarges after
  navigation.

The density calculation is lazy. It is performed only when the density mode
is first enabled and is cached until a new design is loaded.

## File Loading

The open dialog builds its file filter from the reader formats reported by
`pystitch.EmbPattern.supported_formats()`. The viewer therefore follows the
formats supported by the installed pystitch version instead of maintaining a
second hard-coded extension list.

Thread colors are read from the pattern thread list when available. If a file
does not provide thread colors, InkSim uses a deterministic fallback palette.
Embroidery commands such as jumps, color changes, trims, stops, slow, fast,
and end markers are interpreted while the stitch sequence is converted.

## PNG Export

InkSim supports three non-interactive export modes:

```bash
bin/inksim.py design.dst --simple-png output.png
bin/inksim.py design.dst --png shaded-output.png
bin/inksim.py design.dst --icon preview.png
```

Options:

| Option                    | Description                                 |
| ------------------------- | ------------------------------------------- |
| `--simple-png PATH`       | Flat PNG at the design's physical size      |
| `--png PATH`              | Shaded PNG at the design's physical size    |
| `--icon PATH`             | 256 x 256 transparent preview               |
| `--dpi N`                 | DPI for print-sized exports; default is 300 |
| `--bg transparent\|white` | Select the export background                |
| `--grid`                  | Add a 10 mm grid to the exported image      |

Only one export option may be used at a time. Export mode creates a wx
application without entering the interactive main loop, renders the design,
writes PNG metadata, and exits with status 0 on success.

The PNG metadata includes design dimensions, background, layer type, rendering
mode, and DPI where applicable. The interactive fabric/realistic viewport
renderer is intentionally separate from the current standalone export
renderer.

## Performance Notes

- Numba compiles each kernel on its first use; the first render can therefore
  take longer than subsequent renders.
- The viewer caches the rendered bitmap and uses a temporary stretched bitmap
  while zooming, then schedules a full-quality render after zooming settles.
- Pan operations can reuse the cached bitmap without rerendering the stitches.
- The realistic renderer is more expensive than the normal path because it
  evaluates a pixel footprint around every visible stitch and includes a
  separate shadow pass.
- Maximum thread width and maximum sampling steps are bounded to prevent a
  single long stitch from consuming excessive CPU time.

## Design Boundaries and Future Work

InkSim is a preview and inspection tool, not a stitch optimizer or machine
driver. It does not alter the source design during loading and it does not
replace production-specific checks performed by the main Ink/Stitch workflow.

Likely future rendering improvements include:

- continuous satin-surface or normal-map shading;
- better handling of stitch overlap and needle-hole depressions;
- adaptive supersampling for very dense designs;
- optional texture quality controls;
- a shared export/rendering pipeline when visual parity is required.

## Development Checks

Run the syntax check with the project environment:

```bash
python3 -m py_compile bin/inksim.py
```

Check the patch for whitespace errors:

```bash
git diff --check -- bin/inksim.py bin/inksim-readme.md
```
