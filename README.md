# Ballistic Trajectory Simulator

![Tests](https://github.com/anismabaziz/ballistic-trajectory-simulator/actions/workflows/ci.yml/badge.svg)

## Setup

Needs Python 3.11 or newer, which is what the numpy and scipy floors in
`pyproject.toml` require. [uv](https://docs.astral.sh/uv/) handles the
interpreter, the environment, and the lockfile.

Install the project and its test dependencies into `.venv`:

```bash
uv sync
```

`uv.lock` is committed and holds the exact versions the documented physics
results were computed against. `uv sync` installs from it, so a fresh checkout
resolves to the same numbers this README reports.

Run the test suite:

```bash
uv run pytest
```

Run the simulator. `--headless` skips the plotting window and works on a
machine with no display; drop it to watch the plots.

```bash
uv run python main.py --mode target-intercept --headless
```

## Modes

- `target-intercept`: runs stationary target, moving target, and interceptor scenarios
- `real-time-animation`: runs 2D animation with optional target and GIF export
- `three-d-simulation`: runs 3D simulation with optional target, curvature, and salvo
- `interactive-simulator`: runs an interactive close-to-life pygame window

## Headless

`--headless` works in every mode except `interactive-simulator`, which opens a
pygame window and needs a display to run at all. Passing the flag selects the
non-interactive matplotlib backend and skips the plotting window, so the mode
still computes and prints its results. The GIF export path takes the flag too,
since it renders frames without opening a window.

## Examples

Target interception flow example command:

```bash
uv run python main.py --mode target-intercept --launch-speed 300 --headless
```

Real-time animation with moving target example command:

```bash
uv run python main.py --mode real-time-animation --launch-speed 300 --launch-elevation-deg 35 --target-x 2800 --target-velocity-x 40 --target-radius 20
```

Save animation GIF (headless) example command:

```bash
uv run python main.py --mode real-time-animation --output-gif-path trajectory.gif --output-gif-fps 30 --headless
```

3D single trajectory example command:

```bash
uv run python main.py --mode three-d-simulation --launch-speed 300 --launch-elevation-deg 35 --launch-azimuth-deg 10
```

3D salvo example command:

```bash
uv run python main.py --mode three-d-simulation --launch-speed 300 --launch-elevation-deg 35 --enable-salvo --salvo-missile-count 11 --salvo-azimuth-span-deg 40
```

Interactive simulator example command:

```bash
uv run python main.py --mode interactive-simulator --launch-speed 300 --launch-elevation-deg 35 --launch-azimuth-deg 5 --target-x 2800 --target-velocity-x 40 --target-radius 20
```

Interactive controls:

- `SPACE`: launch missile
- `R`: reset simulation
- `UP/DOWN`: elevation angle
- `LEFT/RIGHT`: launch speed
- `Q/E`: azimuth angle
- `T/G`: target speed
- `I/K`: camera pitch up/down
- `J/L`: camera yaw left/right
- `U/O`: camera dolly in/out
- `W/A/S/D`: camera pan on ground plane
- sliders on the right: wind profile, air density, drag coefficient, time scale

## Arguments

- `--mode`: run mode (`target-intercept`, `real-time-animation`, `three-d-simulation`)
- `--launch-speed`: launch speed in m/s
- `--launch-elevation-deg`: launch elevation angle in degrees
- `--launch-azimuth-deg`: launch azimuth angle in degrees (3D mode)
- `--target-x`: initial target X position in meters
- `--target-radius`: target radius in meters
- `--target-velocity-x`: target velocity along X in m/s
- `--frame-interval-ms`: animation frame interval in milliseconds
- `--output-gif-path`: GIF output path for animation
- `--output-gif-fps`: GIF frame rate
- `--headless`: do not open a plotting window, honored by every mode except
  `interactive-simulator`
- `--enable-earth-curvature`: apply Earth curvature correction (3D mode)
- `--enable-salvo`: enable multi-missile azimuth spread (3D mode)
- `--salvo-missile-count`: number of missiles in salvo
- `--salvo-azimuth-span-deg`: total azimuth span across salvo in degrees
