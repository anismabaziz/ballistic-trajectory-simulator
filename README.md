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

Solve a launch angle against a target and report it. `--headless` closes the
figure instead of showing it, so the command runs on a machine with no display.

```bash
uv run python main.py solve --headless
```

Run the real-time renderer:

```bash
uv run python main.py render
```

## Commands

- `solve`: searches for a launch angle that reaches the target and prints the
  angle, whether the shot hit, and the miss distance
- `render`: runs the interactive pygame window

The salvo dispersion and the interceptor scenario are figures, not commands. The
plotting functions for both live in `utils.py` and neither is on the command
line.

## Headless

`--headless` belongs to `solve`, which reports the same thing with or without a
window. It selects the non-interactive matplotlib backend, so the command still
computes and prints its results and still writes a GIF export.

`render` has no such flag. A window is the whole point of it, so passing
`--headless` to `render` is an error.

## Examples

Solve against a stationary target:

```bash
uv run python main.py solve --launch-speed 300 --target-x 3500 --headless
```

Solve against a target walking towards the launcher:

```bash
uv run python main.py solve --target-x 2000 --target-velocity-x 40 --headless
```

Solve with Earth curvature in the model:

```bash
uv run python main.py solve --enable-earth-curvature --headless
```

Export the flight as an animation:

```bash
uv run python main.py solve --output-gif-path trajectory.gif --output-gif-fps 30 --headless
```

Run the renderer against a moving target:

```bash
uv run python main.py render --launch-speed 300 --launch-elevation-deg 35 --launch-azimuth-deg 5 --target-x 2800 --target-velocity-x 40 --target-radius 20
```

Renderer controls:

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

## Auto Solve

The real-time window's Auto Solve button searches for a launch speed, elevation,
and azimuth that hits the target. The search is bounded by a budget in candidate
flights rather than by a stopwatch, so it returns the same solution for the same
input on any machine. The status line under the buttons says how many candidates
it spent and whether that was the whole search or the budget it was given, so a
solution that is the best it found is never read as the best available.

## Arguments

`solve` takes:

- `--launch-speed`: launch speed in m/s
- `--target-x`: initial target X position in meters
- `--target-radius`: target radius in meters
- `--target-velocity-x`: target velocity along X in m/s
- `--enable-earth-curvature`: correct for Earth curvature in the model the angle
  is solved against
- `--output-gif-path`: write the flight out as an animated GIF
- `--output-gif-fps`: GIF frame rate
- `--headless`: do not open a plotting window

`render` takes:

- `--launch-speed`: launch speed in m/s
- `--launch-elevation-deg`: launch elevation angle in degrees
- `--launch-azimuth-deg`: launch azimuth angle in degrees
- `--target-x`: initial target X position in meters
- `--target-radius`: target radius in meters
- `--target-velocity-x`: target velocity along X in m/s
