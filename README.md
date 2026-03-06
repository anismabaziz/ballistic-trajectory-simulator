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
