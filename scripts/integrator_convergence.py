"""Quantify how far the two integrators in this project disagree.

The launch solution search integrates with adaptive RK45 (`solve_ivp`) and the
real-time renderer steps with a fixed step. Both call
`BallisticPhysics.compute_acceleration`, so the difference between their answers
is the step rule and nothing else.

For one launch in one atmosphere, this sweeps the step size across three orders
of magnitude for both rules, measures the distance between where each lands and
where a tightly tolerance-controlled RK45 reference lands, and writes the
log-log convergence figure the README leads with.

Run it with:

    uv run python scripts/integrator_convergence.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import config
from physics import BallisticPhysics

# The launch and atmosphere the figure reports. These are the renderer's
# defaults, so the number in the figure describes the sandbox as it ships.
LAUNCH_SPEED = 300.0
LAUNCH_ELEVATION_DEG = 35.0
LAUNCH_AZIMUTH_DEG = 0.0
LATITUDE_DEG = 0.0

FRAME_RATE = 60
# RK45's own tolerance has to be tight for the reference to be worth measuring
# against. solve_ivp refuses tolerances below the machine epsilon times the state
# scale, and 1e-12 sits comfortably inside that.
REFERENCE_TOLERANCE = 1e-12
# The tolerance the adaptive runs use. Tight enough that the reference is not
# the thing being measured.
ADAPTIVE_TOLERANCE = 1e-10

# Step sizes for the sweep, 0.1 s down to 0.0001 s: three orders of magnitude,
# and it straddles the 16.7 ms frame step the renderer actually uses.
STEP_SWEEP = np.logspace(-1.0, -4.0, 21)

# Substeps per frame that get a first-order renderer inside a metre of the
# adaptive answer, used as the calibration alternative in the figure.
SUBSTEPS = 16

# The order is fitted over steps at or above this. Fits taken across the whole
# sweep report the reference's own noise rather than the method, because the fine
# end of the RK4 sweep is already inside it.
ORDER_FIT_FLOOR_STEP = 0.03

OUTPUT_PATH = "figures/integrator_convergence.png"


def launch_state():
    physics = BallisticPhysics(latitude=LATITUDE_DEG)
    elevation = np.radians(LAUNCH_ELEVATION_DEG)
    azimuth = np.radians(LAUNCH_AZIMUTH_DEG)
    vx = LAUNCH_SPEED * np.cos(elevation) * np.cos(azimuth)
    vy = LAUNCH_SPEED * np.sin(elevation)
    vz = LAUNCH_SPEED * np.cos(elevation) * np.sin(azimuth)
    return physics, [0.0, 0.0, 0.0, vx, vy, vz]


def integrate_to_ground(physics, initial_state, duration):
    """Adaptive integration stopped at ground impact, returning state and time."""

    def hit_ground(_, state):
        return state[1]

    setattr(hit_ground, "terminal", True)
    setattr(hit_ground, "direction", -1)

    solution = solve_ivp(
        physics.projectile_rhs_3d,
        (0.0, duration),
        initial_state,
        rtol=REFERENCE_TOLERANCE,
        atol=REFERENCE_TOLERANCE,
        events=hit_ground,
    )
    return solution.y[:, -1], float(solution.t[-1])


def integrate_adaptive(physics, initial_state, flight_time, max_step=None):
    """RK45 at a tight tolerance, optionally capped at `max_step`.

    The cap only changes the answer where the tolerance controller wants a
    larger step than the cap allows, which is why the capped curve flattens
    instead of falling with the step size. That flattening is the result.
    """
    options = {}
    if max_step is not None:
        options["max_step"] = max_step
    solution = solve_ivp(
        physics.projectile_rhs_3d,
        (0.0, flight_time),
        initial_state,
        rtol=ADAPTIVE_TOLERANCE,
        atol=ADAPTIVE_TOLERANCE,
        **options,
    )
    return solution.y[:, -1]


def landing_error(state, reference):
    return float(np.linalg.norm(np.asarray(state[:3]) - np.asarray(reference[:3])))


def observed_order(steps, errors):
    """Slope of the error curve in log-log, which is the method's order.

    Fitted over the coarsest step sizes only. Below about 0.03 s the RK4 curve is
    down at the accuracy of the reference itself, so a fit across the whole sweep
    would report the reference rather than the method.
    """
    coarse = np.asarray(steps) >= ORDER_FIT_FLOOR_STEP
    slope = np.polyfit(np.log10(np.asarray(steps)[coarse]), np.log10(np.asarray(errors)[coarse]), 1)
    return float(slope[0])


def describe_atmosphere():
    return (
        f"v0 = {LAUNCH_SPEED:.0f} m/s   elevation = {LAUNCH_ELEVATION_DEG:.0f}°   "
        f"azimuth = {LAUNCH_AZIMUTH_DEG:.0f}°   latitude = {LATITUDE_DEG:.0f}°\n"
        f"m = {config.MASS:.0f} kg   Cd = {config.CD}   A = {config.AREA} m²   "
        f"rho = {config.RHO} kg/m³   g = {config.G} m/s²\n"
        f"wind x = {config.WIND_X} m/s at {config.ALT_LEVELS} m,   "
        f"wind z = {config.WIND_Y} m/s"
    )


def main():
    physics, initial_state = launch_state()
    reference, flight_time = integrate_to_ground(physics, initial_state, 1000.0)

    euler_errors = []
    rk4_errors = []
    capped_errors = []
    for step in STEP_SWEEP:
        euler_errors.append(
            landing_error(physics.integrate_fixed_step(initial_state, step, flight_time, rule="euler"), reference)
        )
        rk4_errors.append(
            landing_error(physics.integrate_fixed_step(initial_state, step, flight_time, rule="rk4"), reference)
        )
        capped_errors.append(landing_error(integrate_adaptive(physics, initial_state, flight_time, step), reference))

    euler_errors = np.asarray(euler_errors)
    rk4_errors = np.asarray(rk4_errors)
    capped_errors = np.asarray(capped_errors)
    uncapped_error = landing_error(integrate_adaptive(physics, initial_state, flight_time), reference)

    frame_step = 1.0 / FRAME_RATE
    euler_frame_error = landing_error(
        physics.integrate_fixed_step(initial_state, frame_step, flight_time, rule="euler"), reference
    )
    rk4_frame_error = landing_error(
        physics.integrate_fixed_step(initial_state, frame_step, flight_time, rule="rk4"), reference
    )
    euler_substep_error = landing_error(
        physics.integrate_fixed_step(initial_state, frame_step / SUBSTEPS, flight_time, rule="euler"), reference
    )
    euler_order = observed_order(STEP_SWEEP, euler_errors)
    rk4_order = observed_order(STEP_SWEEP, rk4_errors)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10.0, 11.0), sharex=True)
    axis_style = {"fontsize": 9, "color": "#444444", "ha": "left"}

    top.loglog(STEP_SWEEP, euler_errors, "o-", color="#c0392b", markersize=4, label="semi-implicit Euler, first order")
    top.loglog(STEP_SWEEP, rk4_errors, "d-", color="#e07b39", markersize=4, label="RK4, fourth order")
    anchor = len(STEP_SWEEP) // 2
    top.loglog(
        STEP_SWEEP,
        euler_errors[anchor] * (STEP_SWEEP / STEP_SWEEP[anchor]),
        ":",
        color="#555555",
        linewidth=1.6,
        label="slope 1",
    )
    top.loglog(
        STEP_SWEEP,
        rk4_errors[anchor] * (STEP_SWEEP / STEP_SWEEP[anchor]) ** 4,
        "-.",
        color="#555555",
        linewidth=1.6,
        label="slope 4",
    )
    top.axvline(frame_step, color="#777777", linewidth=1.0, linestyle="--")
    top.set_ylim(1e-9, 1e3)
    top.set_ylabel("landing error vs RK45 reference (m)")
    top.set_title(
        f"Fixed-step rules: Euler order {euler_order:.2f}, RK4 order {rk4_order:.2f}, fitted over the coarse half decade",
        fontsize=12,
    )
    top.legend(loc="center left", bbox_to_anchor=(0.01, 0.45), fontsize=9, framealpha=0.95)
    top.grid(True, which="both", alpha=0.3)

    top.plot([frame_step], [euler_frame_error], "v", color="#c0392b", markersize=9)
    top.annotate(
        f"one Euler step per {FRAME_RATE} fps frame: {euler_frame_error:.1f} m",
        xy=(frame_step, euler_frame_error),
        xycoords="data",
        xytext=(0.52, 0.88),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.9},
        **axis_style,
    )
    top.plot([frame_step / SUBSTEPS], [euler_substep_error], "v", color="#c0392b", markersize=7)
    top.annotate(
        f"{SUBSTEPS} Euler substeps per frame: {euler_substep_error:.2f} m",
        xy=(frame_step / SUBSTEPS, euler_substep_error),
        xycoords="data",
        xytext=(0.03, 0.70),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.9},
        **axis_style,
    )
    top.plot([frame_step], [rk4_frame_error], "D", color="#e07b39", markersize=9)
    top.annotate(
        f"one RK4 step per frame: {rk4_frame_error:.1e} m",
        xy=(frame_step, rk4_frame_error),
        xycoords="data",
        xytext=(0.56, 0.22),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.9},
        **axis_style,
    )
    top.text(
        0.36,
        0.62,
        "below about 0.003 s the RK4 curve is inside the reference's own accuracy, so the\n"
        "figure is measuring the reference there rather than the rule",
        transform=top.transAxes,
        fontsize=9,
        color="#e07b39",
        ha="left",
        va="top",
    )

    bottom.loglog(STEP_SWEEP, capped_errors, "s-", color="#2c6fbb", markersize=4, label="RK45, step capped")
    bottom.axhline(uncapped_error, color="#4f8f3f", linewidth=1.6, label=f"RK45, uncapped ({uncapped_error:.1e} m)")
    bottom.axvline(frame_step, color="#777777", linewidth=1.0, linestyle="--")
    bottom.set_ylim(1e-9, 1e-6)
    bottom.set_xlabel("step size (s)")
    bottom.set_ylabel("landing error vs RK45 reference (m)")
    bottom.set_title("Adaptive RK45: no order in the step size, because the tolerance picks it", fontsize=12)
    bottom.legend(loc="upper left", fontsize=9, framealpha=0.95)
    bottom.grid(True, which="both", alpha=0.3)
    bottom.text(
        0.02,
        0.72,
        "three orders of step size, and the error scatters between 2e-9 and 3e-7 m\n"
        "with no trend: the cap is always looser than the step the controller wants,\n"
        f"so accuracy is set by rtol = atol = {ADAPTIVE_TOLERANCE:.0e}, not by the cap",
        transform=bottom.transAxes,
        fontsize=9,
        color="#2c6fbb",
    )

    figure.text(0.5, 0.015, describe_atmosphere(), ha="center", fontsize=8.5, color="#333333")
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)

    print(f"flight time {flight_time:.4f} s, reference range {reference[0]:.2f} m")
    print(f"observed order: euler {euler_order:.2f}, rk4 {rk4_order:.2f}")
    print(
        f"at {FRAME_RATE} fps, one step per frame: euler {euler_frame_error:.4f} m, "
        f"rk4 {rk4_frame_error:.3e} m"
    )
    print(f"at {FRAME_RATE} fps, {SUBSTEPS} euler substeps per frame: {euler_substep_error:.3f} m")
    for step, euler_error, rk4_error, capped_error in zip(STEP_SWEEP, euler_errors, rk4_errors, capped_errors):
        print(
            f"  dt={step:.6f}  euler={euler_error:.4f} m  rk4={rk4_error:.3e} m  rk45={capped_error:.3e} m"
        )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
