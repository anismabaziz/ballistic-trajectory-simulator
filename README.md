# Ballistic Trajectory Simulator

Run the simulator with the project virtual environment:

```bash
./venv/bin/python main.py --mode target-intercept
```

## Modes

- `target-intercept`: runs stationary target, moving target, and interceptor scenarios
- `real-time-animation`: runs 2D animation with optional target and GIF export
- `three-d-simulation`: runs 3D simulation with optional target, curvature, and salvo

## Examples

Target interception flow:

```bash
./venv/bin/python main.py --mode target-intercept --launch-speed 300
```

Real-time animation with moving target:

```bash
./venv/bin/python main.py --mode real-time-animation --launch-speed 300 --launch-elevation-deg 35 --target-x 2800 --target-velocity-x 40 --target-radius 20
```

Save animation GIF (headless):

```bash
./venv/bin/python main.py --mode real-time-animation --output-gif-path trajectory.gif --output-gif-fps 30 --headless
```

3D single trajectory:

```bash
./venv/bin/python main.py --mode three-d-simulation --launch-speed 300 --launch-elevation-deg 35 --launch-azimuth-deg 10
```

3D salvo:

```bash
./venv/bin/python main.py --mode three-d-simulation --launch-speed 300 --launch-elevation-deg 35 --enable-salvo --salvo-missile-count 11 --salvo-azimuth-span-deg 40
```

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
- `--headless`: do not open plotting window
- `--enable-earth-curvature`: apply Earth curvature correction (3D mode)
- `--enable-salvo`: enable multi-missile azimuth spread (3D mode)
- `--salvo-missile-count`: number of missiles in salvo
- `--salvo-azimuth-span-deg`: total azimuth span across salvo in degrees

If `python3 main.py` fails with `ModuleNotFoundError` (e.g., `numpy`), use `./venv/bin/python`.
