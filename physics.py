import numpy as np
from scipy.integrate import solve_ivp
from typing import overload, Literal
import config

def projectile_rhs_3d(t, Y, m, g, alt_levels, wind_x_vals, wind_y_vals, latitude=0):
    x, y, z, vx, vy, vz = Y
    wind_x = np.interp(y, alt_levels, wind_x_vals)
    wind_y = np.interp(y, alt_levels, wind_y_vals)
    vx_rel = vx - wind_x
    vy_rel = vy
    vz_rel = vz - wind_y
    v_rel = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

    if v_rel != 0:
        Fd = 0.5 * config.RHO * config.CD * config.AREA * v_rel**2
        ax = -(Fd / m) * (vx_rel / v_rel)
        ay = -g - (Fd / m) * (vy_rel / v_rel)
        az = -(Fd / m) * (vz_rel / v_rel)
    else:
        ax, ay, az = 0.0, -g, 0.0

    # Coriolis effect
    OMEGA = 7.2921e-5
    ax += 2 * OMEGA * vz * np.sin(latitude)
    az += -2 * OMEGA * vx * np.sin(latitude)
    return [vx, vy, vz, ax, ay, az]


@overload
def trajectory_3d(v0, angle_deg, m=config.MASS, g=config.G,
                  alt_levels=None, wind_x_vals=None, wind_y_vals=None,
                  latitude=0, return_time: Literal[False] = False,
                  max_step: float = 0.05):
    ...


@overload
def trajectory_3d(v0, angle_deg, m=config.MASS, g=config.G,
                  alt_levels=None, wind_x_vals=None, wind_y_vals=None,
                  latitude=0, return_time: Literal[True] = True,
                  max_step: float = 0.05):
    ...


def trajectory_3d(v0, angle_deg, m=config.MASS, g=config.G,
                  alt_levels=None, wind_x_vals=None, wind_y_vals=None,
                  latitude=0, return_time=False, max_step=0.05):
    if alt_levels is None:
        alt_levels = config.ALT_LEVELS
    if wind_x_vals is None:
        wind_x_vals = config.WIND_X
    if wind_y_vals is None:
        wind_y_vals = config.WIND_Y

    a = np.radians(angle_deg)
    vx0 = v0 * np.cos(a)
    vy0 = v0 * np.sin(a)
    vz0 = 0.0
    Y0 = [0.0, 0.0, 0.0, vx0, vy0, vz0]

    def hit_ground(t, Y):
        return Y[1]
    setattr(hit_ground, "terminal", True)
    setattr(hit_ground, "direction", -1)

    sol = solve_ivp(
        lambda t, Y: projectile_rhs_3d(t, Y, m, g, alt_levels, wind_x_vals, wind_y_vals, latitude),
        (0, 1000),
        Y0,
        max_step=max_step,
        events=hit_ground
    )

    xs, ys, zs = sol.y[0], sol.y[1], sol.y[2]
    R, T, H = xs[-1], sol.t[-1], np.max(ys)
    if return_time:
        return xs, ys, zs, sol.t, R, T, H
    return xs, ys, zs, R, T, H
