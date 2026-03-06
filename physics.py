import numpy as np
from scipy.integrate import solve_ivp
import config

def projectile_rhs_3d(t, Y, m, g, alt_levels, wind_x_vals, wind_y_vals, latitude=0):
    """
    Right-hand side for 3D projectile motion with drag, wind, and optional Coriolis effect.
    
    Y = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = Y

    # Interpolate wind at current altitude
    wind_x = np.interp(y, alt_levels, wind_x_vals)
    wind_y = np.interp(y, alt_levels, wind_y_vals)

    # Relative velocity for drag
    vx_rel = vx - wind_x
    vy_rel = vy         # vertical unaffected by horizontal wind
    vz_rel = vz - wind_y
    v_rel = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

    # Drag acceleration
    if v_rel != 0:
        Fd = 0.5 * config.RHO * config.CD * config.AREA * v_rel**2
        ax = -(Fd / m) * (vx_rel / v_rel)
        ay = -g - (Fd / m) * (vy_rel / v_rel)
        az = -(Fd / m) * (vz_rel / v_rel)
    else:
        ax, ay, az = 0.0, -g, 0.0

    # simplified Coriolis effect (x-z plane)
    OMEGA = 7.2921e-5  # rad/s
    ax += 2 * OMEGA * vz * np.sin(latitude)
    az += -2 * OMEGA * vx * np.sin(latitude)

    return [vx, vy, vz, ax, ay, az]


def trajectory_3d(v0, angle_deg, m=config.MASS, g=config.G,
                  alt_levels=None, wind_x_vals=None, wind_y_vals=None,
                  latitude=0):
    """
    Compute 3D trajectory of a projectile with drag, wind, and optional Coriolis effect.
    
    Returns:
        xs, ys, zs: positions arrays
        R: horizontal range
        T: time of flight
        H: maximum height
    """
    # Use defaults if not provided
    if alt_levels is None:
        alt_levels = config.ALT_LEVELS
    if wind_x_vals is None:
        wind_x_vals = config.WIND_X
    if wind_y_vals is None:
        wind_y_vals = config.WIND_Y

    # Initial velocities
    a = np.radians(angle_deg)
    vx0 = v0 * np.cos(a)
    vy0 = v0 * np.sin(a)
    vz0 = 0.0  # no lateral velocity initially

    Y0 = [0.0, 0.0, 0.0, vx0, vy0, vz0]

    # Event: stop integration when projectile hits ground
    def hit_ground(t, Y):
        return Y[1]  # y-coordinate
    hit_ground.terminal = True
    hit_ground.direction = -1  # only downward crossing

    # Integrate using solve_ivp
    sol = solve_ivp(
        fun=lambda t, Y: projectile_rhs_3d(t, Y, m, g, alt_levels, wind_x_vals, wind_y_vals, latitude),
        t_span=(0, 1000),  # maximum integration time
        y0=Y0,
        max_step=0.01,
        events=hit_ground
    )

    # Extract results
    x, y, z, vx, vy, vz = sol.y
    xs, ys, zs = np.array(x), np.array(y), np.array(z)

    T = sol.t[-1]         # total time of flight
    R = xs[-1]            # horizontal range
    H = np.max(ys)        # maximum height

    return xs, ys, zs, R, T, H