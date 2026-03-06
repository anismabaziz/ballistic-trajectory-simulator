# Part 1: Basic Projectile Motion

## What is a Projectile?

A projectile is any object that is thrown or shot into the air and moves under the influence of gravity only.

After a projectile is launched, ignoring air and other factors, only gravity acts upon it making it go down.

## Motion in 2 Directions

A projectile moves in 2 directions at once:

- **Horizontally**: on the x-axis
- **Vertically**: on the y-axis

Horizontal and vertical motion act independently but they share time.

## Horizontal Motion Equation

**Equation:** $x(t) = v \cdot \cos(\theta) \cdot t$

| Symbol         | Description                               |
| -------------- | ----------------------------------------- |
| $v$            | Speed at which the projectile is launched |
| $t$            | Time since launch                         |
| $\theta$       | Launch angle                              |
| $\cos(\theta)$ | Horizontal component of velocity          |

&gt; Horizontal speed doesn't slow down since we are ignoring air resistance.

## Vertical Motion Equation

**Equation:** $y(t) = v \cdot \sin(\theta) \cdot t - \frac{1}{2}gt^2$

| Symbol             | Description                                          |
| ------------------ | ---------------------------------------------------- |
| $v$                | Speed at which the projectile is launched            |
| $t$                | Time since launch                                    |
| $\theta$           | Launch angle                                         |
| $\sin(\theta)$     | Vertical component of velocity (how fast it goes up) |
| $-\frac{1}{2}gt^2$ | Gravity pulling it down                              |

## Time of Flight

**Formula:** $T = \frac{2v \cdot \sin(\theta)}{g}$

This comes from when $y(t) = 0$, meaning the projectile lands back on the ground.

## Maximum Horizontal Range

**Formula:** $R = \frac{v^2 \cdot \sin(2\theta)}{g}$

This calculates how far the projectile goes horizontally before hitting the ground. We can also get it from horizontal speed × time taken.

$\sin(2\theta)$ comes from combining horizontal and vertical speed.

**Maximum range is achieved when $\theta = 45°$**

---

# Part 2: Add Interactivity to the Project

## Optimum Angle Finder

The optimal angle is the one that achieves the highest horizontal range.

Max horizontal range is achieved when $\sin(2\theta) = 1 \Rightarrow 2\theta = 90°$

This means **$\theta = 45°$**

---

# Part 3: Air Resistance and Drag Physics

## Reasons for Switch

In a vacuum, projectiles follow a perfect parabola because the only acting force upon them is gravity.

In air, projectiles experience **drag** which is an opposite force to their motion, meaning both vertical and horizontal velocity decrease over time.

In the first phase of our simulation we only had closed-form equations which only work in closed environments without air.

With drag, acceleration depends on velocity, so we can't just solve algebraically.

**Solution:** We use numerical integration

- We take small steps $dt$
- We update velocity and position iteratively

This is done through methods like: **Euler method**, **RK4 (Runge-Kutta 4th order)**

## Drag Force Equation

$$F_d = 0.5 \cdot \rho \cdot C_d \cdot A \cdot v^2$$

| Symbol | Description                                                        |
| ------ | ------------------------------------------------------------------ |
| $F_d$  | Magnitude of drag (N)                                              |
| $\rho$ | Air density (kg/m³) → usually **1.225** at sea level               |
| $C_d$  | Drag coefficient (depends on shape) → **0.47** for a sphere        |
| $A$    | Cross-sectional area (m²) — how big the missile is "from the side" |
| $v$    | Speed of the projectile (m/s)                                      |

&gt; **Key idea:** Drag increases with speed **squared**, so faster projectiles feel _much_ more drag.

## Net Accelerations

The drag force always points **opposite** the velocity vector.

### Horizontal Acceleration:

$$a_x = -\frac{F_d}{m} \cdot \frac{v_x}{|v|}$$

### Vertical Acceleration:

$$a_y = -g - \frac{F_d}{m} \cdot \frac{v_y}{|v|}$$

### Where:

- $\frac{v_x}{|v|}$ and $\frac{v_y}{|v|}$ are components of the **unit vector** of velocity
- $-\frac{F_d}{m}$ scales the drag into an acceleration
- $-g$ acts only vertically

- now we move from noral euleur method to using RK4 then we implement RK4 using bulting scipy functions rather than using loops for updates

---

# Part 4: Physics Engine Refactor (Class-Based)

The simulator now uses an object-oriented physics engine in `physics.py`.

## `BallisticPhysics` class

`BallisticPhysics` stores the full environment and physical model:

- mass, gravity, air density, drag coefficient, cross-sectional area
- altitude-dependent wind tables
- optional latitude for Coriolis effects

Main methods:

- `projectile_rhs_3d(t, state)` computes the differential equations
- `trajectory_3d(v0, angle_deg, ...)` integrates with `scipy.integrate.solve_ivp`

This makes it easier to run multiple scenarios with different environments by creating multiple simulator instances.

Example:

```python
from physics import BallisticPhysics

sim = BallisticPhysics()
xs, ys, zs, t, R, T, H = sim.trajectory_3d(300, 35, return_time=True)
```

Backward compatibility is preserved through a module-level wrapper function:

```python
from physics import trajectory_3d
```

---

# Part 5: Target Interception and Collision Detection

Phase 5 features are now implemented end-to-end.

## New capabilities

1. **Stationary target collision check**
   - Place a target at `(x, y, z)` with radius `r`
   - Detect hit by checking missile-target distance at each simulation time

2. **Hit/miss visualization**
   - Green marker/circle on hit
   - Red miss marker on miss
   - Closest-approach point highlighted

3. **Angle solver for interception**
   - Uses root finding (`scipy.optimize.brentq`) for stationary target range matching

4. **Moving target support**
   - Targets can move with constant velocity `(vx, vy, vz)`
   - Solver searches launch angle minimizing miss distance

5. **Interceptor missile scenario**
   - Simulate second missile launched from origin
   - Solve launch angle minimizing separation from primary missile

6. **Closest-approach metric**
   - Reports minimum Euclidean distance over shared simulation time

## Modules added/updated

- `targets.py`
  - `Target` class (stationary or moving targets)
  - `check_collision(...)`
  - `closest_approach_between_trajectories(...)`

- `utils.py`
  - plotting helpers for hit/miss and intercept visualization
  - `find_launch_angle(...)`
  - `solve_moving_target_angle(...)`
  - `solve_interceptor_angle(...)`

- `main.py`
  - runs phase scenarios:
    - stationary target
    - moving target
    - interceptor missile

## Run

```bash
python3 main.py
```

If you are using the local virtual environment:

```bash
./venv/bin/python main.py
```

---

# Part 6: Real-Time Animation

Phase 6 Option A is implemented using Matplotlib `FuncAnimation`.

## Implemented items

- **6A.1 Pre-compute trajectory**: simulate first, store `x[i], y[i]`
- **6A.2 Animate point + trail**: moving missile dot + growing trail
- **6A.3 Speed control**: frame speed via `interval_ms`
- **6A.4 Save to GIF**: optional GIF export with Pillow writer

## Run animation

```bash
./venv/bin/python main.py --mode animate --velocity 300 --angle 35 --interval-ms 30
```

## Save GIF

```bash
./venv/bin/python main.py --mode animate --save-gif trajectory.gif --gif-fps 30 --no-show
```

## Keep using Phase 5 flow

```bash
./venv/bin/python main.py --mode phase5
```

## Dependency note

If `python3 main.py` fails with `ModuleNotFoundError: No module named 'numpy'`, use the project virtual environment:

```bash
./venv/bin/python main.py
```
