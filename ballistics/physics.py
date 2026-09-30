import numpy as np
from scipy.integrate import solve_ivp
from typing import Literal, overload
from . import config


class BallisticPhysics:
    OMEGA = 7.2921e-5

    def __init__(
        self,
        mass=config.MASS,
        gravity=config.G,
        rho=config.RHO,
        drag_coefficient=config.CD,
        area=config.AREA,
        alt_levels=None,
        wind_x_vals=None,
        wind_z_vals=None,
        wind_y_vals=None,
        wind_vertical_vals=None,
        latitude=0.0,
    ):
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.rho = float(rho)
        self.drag_coefficient = float(drag_coefficient)
        self.area = float(area)
        self.latitude = float(latitude)
        self.alt_levels = np.asarray(alt_levels if alt_levels is not None else config.ALT_LEVELS, dtype=float)
        self.wind_x_vals = np.asarray(wind_x_vals if wind_x_vals is not None else config.WIND_X, dtype=float)
        wind_z_source = wind_z_vals if wind_z_vals is not None else wind_y_vals
        self.wind_z_vals = np.asarray(wind_z_source if wind_z_source is not None else config.WIND_Y, dtype=float)
        if wind_vertical_vals is None:
            self.wind_vertical_vals = np.zeros_like(self.alt_levels, dtype=float)
        else:
            self.wind_vertical_vals = np.asarray(wind_vertical_vals, dtype=float)

    def projectile_rhs_3d(self, _, state):
        _, y, _, vx, vy, vz = state
        ax, ay, az = self.compute_acceleration(y, vx, vy, vz)
        return [vx, vy, vz, ax, ay, az]

    def integrate_fixed_step(self, initial_state, step, duration, rule):
        """Advance the state through the same acceleration model, fixed step.

        The renderer steps this way so its frame pacing stays deterministic: it
        knows before the frame starts how many steps it will take, and how long
        each one costs. Two rules are available and the convergence study
        measures both against the adaptive solver. `rk4` is the classical
        four-stage rule and is what the renderer uses. `euler` is semi-implicit
        Euler, the rule the renderer shipped with, kept because it is the
        baseline the study compares against and it is first order.
        """
        if rule not in FIXED_STEP_RULES:
            raise ValueError(f"unknown fixed-step rule: {rule!r}")
        # Copy, because the Euler step updates the state in place and a caller
        # handing us its own array should not find it changed underneath.
        state = np.array(initial_state, dtype=float)
        count = max(1, int(np.ceil(duration / step)))
        # Trim the step so the run lands exactly on `duration`. Without this a
        # sweep over step sizes compares runs that stopped at different times,
        # and the error looks noisier than the rule is.
        step = duration / count
        for _ in range(count):
            state = FIXED_STEP_RULES[rule](self, state, step)
        return state

    def _euler_step(self, state, step):
        ax, ay, az = self.compute_acceleration(state[1], state[3], state[4], state[5])
        state[3] += ax * step
        state[4] += ay * step
        state[5] += az * step
        state[0] += state[3] * step
        state[1] += state[4] * step
        state[2] += state[5] * step
        return state

    def _rk4_step(self, state, step):
        def derivative(point):
            return np.asarray(self.projectile_rhs_3d(0.0, point), dtype=float)

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * step * k1)
        k3 = derivative(state + 0.5 * step * k2)
        k4 = derivative(state + step * k3)
        return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def compute_acceleration(self, altitude_y, vx, vy, vz):
        y = altitude_y
        wind_x = np.interp(y, self.alt_levels, self.wind_x_vals)
        wind_z = np.interp(y, self.alt_levels, self.wind_z_vals)
        wind_vertical = np.interp(y, self.alt_levels, self.wind_vertical_vals)

        vx_rel = vx - wind_x
        vy_rel = vy - wind_vertical
        vz_rel = vz - wind_z
        v_rel = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

        if v_rel != 0:
            drag_force = 0.5 * self.rho * self.drag_coefficient * self.area * v_rel**2
            ax = -(drag_force / self.mass) * (vx_rel / v_rel)
            ay = -self.gravity - (drag_force / self.mass) * (vy_rel / v_rel)
            az = -(drag_force / self.mass) * (vz_rel / v_rel)
        else:
            ax, ay, az = 0.0, -self.gravity, 0.0

        ax += 2 * self.OMEGA * vz * np.sin(self.latitude)
        az += -2 * self.OMEGA * vx * np.sin(self.latitude)
        return ax, ay, az

    @overload
    def trajectory_3d(
        self,
        v0,
        angle_deg,
        azimuth_deg=0.0,
        return_time: Literal[False] = False,
        max_step=0.05,
        t_final=1000.0,
        apply_earth_curvature=False,
        earth_radius=6_371_000.0,
    ):
        ...

    @overload
    def trajectory_3d(
        self,
        v0,
        angle_deg,
        azimuth_deg=0.0,
        return_time: Literal[True] = True,
        max_step=0.05,
        t_final=1000.0,
        apply_earth_curvature=False,
        earth_radius=6_371_000.0,
    ):
        ...

    def trajectory_3d(
        self,
        v0,
        angle_deg,
        azimuth_deg=0.0,
        return_time=False,
        max_step=0.05,
        t_final=1000.0,
        apply_earth_curvature=False,
        earth_radius=6_371_000.0,
    ):
        launch_angle = np.radians(angle_deg)
        azimuth = np.radians(azimuth_deg)
        vx0 = v0 * np.cos(launch_angle) * np.cos(azimuth)
        vy0 = v0 * np.sin(launch_angle)
        vz0 = v0 * np.cos(launch_angle) * np.sin(azimuth)
        initial_state = [0.0, 0.0, 0.0, vx0, vy0, vz0]

        def hit_ground(_, state):
            return state[1]

        setattr(hit_ground, "terminal", True)
        setattr(hit_ground, "direction", -1)

        solution = solve_ivp(
            self.projectile_rhs_3d,
            (0.0, t_final),
            initial_state,
            max_step=max_step,
            events=hit_ground,
        )

        xs, ys, zs = solution.y[0], solution.y[1], solution.y[2]
        if apply_earth_curvature:
            ground_range = np.sqrt(xs**2 + zs**2)
            curvature_drop = (ground_range**2) / (2.0 * earth_radius)
            ys = ys - curvature_drop

        R, T, H = xs[-1], solution.t[-1], np.max(ys)
        if return_time:
            return xs, ys, zs, solution.t, R, T, H
        return xs, ys, zs, R, T, H



FIXED_STEP_RULES = {"euler": BallisticPhysics._euler_step, "rk4": BallisticPhysics._rk4_step}
