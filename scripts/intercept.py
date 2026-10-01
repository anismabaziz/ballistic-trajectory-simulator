"""An interceptor launched at a primary that is already in flight.

Both projectiles leave the same launcher, so scored from launch the smallest
separation is zero at t = 0 for every angle the search could pick. The figure
therefore waits: the interceptor is launched `LAUNCH_DELAY_S` after the primary,
and the search runs against the primary's remaining flight. What the figure draws
is a chase, not a shared start line.

The angle comes from the interceptor search in `ballistics.searches`, over the
same acceleration model everything else uses. The search scores 190 grid angles
on a coarse step, so the angle it names is a ranking rather than a promise. The
figure flies that one angle again at `REPLAY_MAX_STEP_S` and draws the replayed
flight, which is what keeps the picture and the reported miss distance the same
claim rather than two numbers a few centimetres apart.

Run it with:

    uv run python scripts/intercept.py
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ballistics.physics import BallisticPhysics
from ballistics.searches import solve_interceptor_angle
from ballistics.targets import miss_distance_between_trajectories
from figure_common import describe_atmosphere, run

# The primary is the shot the renderer defaults to. At 5 s it is about 1150 m
# downrange and 690 m up, far enough ahead that the interceptor has a flight of
# its own to plan.
PRIMARY_SPEED_MPS = 300.0
PRIMARY_ELEVATION_DEG = 35.0
PRIMARY_MAX_STEP_S = 0.05

# 500 m/s is the speed at which the search converges on a couple of metres. The
# same geometry at 320 m/s stalls near 70 m, which is why this is not the speed
# the renderer throws.
INTERCEPTOR_SPEED_MPS = 500.0
LAUNCH_DELAY_S = 5.0

# The step the named angle is replayed at. A tenth of the step the search scored
# its grid on, so the miss distance the figure reports is the integrator's answer
# rather than a ranking aid's.
REPLAY_MAX_STEP_S = 0.01

OUTPUT_PATH = "figures/intercept.png"


@dataclass(frozen=True)
class PrimaryFlight:
    """The primary's whole flight, and the part of it the interceptor chases."""

    # The whole flight, so the figure can draw what was already in the air before
    # the interceptor launched.
    full: np.ndarray
    full_time: np.ndarray
    # From `LAUNCH_DELAY_S` on, with the clock restarted at zero, which is the
    # slice both flights are compared over.
    in_flight: np.ndarray
    in_flight_time: np.ndarray


@dataclass(frozen=True)
class Intercept:
    """The solution, the flight it was checked against, and the miss it made."""

    angle: float
    miss_distance: float
    miss_time: float
    interceptor: np.ndarray
    interceptor_time: np.ndarray


def primary_in_flight(physics):
    """The primary from `LAUNCH_DELAY_S` on, with its clock restarted at zero.

    The slice is what makes the miss distance mean anything. Both projectiles
    leave one launcher at the same instant, so the smallest separation over their
    whole flight is the origin at t = 0 whatever angle the interceptor takes.
    Starting the shared clock once the primary is under way puts the two in
    different places at the new zero.
    """
    xs, ys, zs, t_array, _range, _flight_time, _peak = physics.trajectory_3d(
        PRIMARY_SPEED_MPS,
        PRIMARY_ELEVATION_DEG,
        return_time=True,
        max_step=PRIMARY_MAX_STEP_S,
    )
    full = np.column_stack((xs, ys, zs))
    start = int(np.searchsorted(t_array, LAUNCH_DELAY_S))
    return PrimaryFlight(
        full=full,
        full_time=t_array,
        in_flight=full[start:],
        in_flight_time=t_array[start:] - t_array[start],
    )


def search_intercept(physics, primary):
    """The angle the interceptor search names for this geometry."""
    solution = solve_interceptor_angle(
        primary.in_flight,
        primary.in_flight_time,
        INTERCEPTOR_SPEED_MPS,
        trajectory_func=physics.trajectory_3d,
    )
    return solution["angle"]


def replay_intercept(physics, primary, angle):
    """Fly the named angle again, tightly, and measure the miss against it.

    The search's own distance is the best of 190 grid samples, which is a ranking
    aid rather than a promise. The replayed flight is the one the figure draws and
    the one the reported miss distance is measured on, so the marker on the picture
    sits on the curve the picture shows.
    """
    xs, ys, zs, t_array, _range, _flight_time, _peak = physics.trajectory_3d(
        INTERCEPTOR_SPEED_MPS,
        angle,
        return_time=True,
        max_step=REPLAY_MAX_STEP_S,
    )
    interceptor = np.column_stack((xs, ys, zs))
    miss_distance, miss_time, _index = miss_distance_between_trajectories(
        primary.in_flight, interceptor, primary.in_flight_time, t_array
    )
    return Intercept(
        angle=angle,
        miss_distance=float(miss_distance),
        miss_time=float(miss_time),
        interceptor=interceptor,
        interceptor_time=t_array,
    )


def solve(physics):
    """The interceptor's solution, and the primary it was found against."""
    primary = primary_in_flight(physics)
    return primary, replay_intercept(physics, primary, search_intercept(physics, primary))


def position_at(trajectory, times, when):
    """Where a flight was at a given time on its own clock, interpolated."""
    return np.array(
        [
            np.interp(when, times, trajectory[:, 0]),
            np.interp(when, times, trajectory[:, 1]),
            np.interp(when, times, trajectory[:, 2]),
        ]
    )


def separation_over_time(primary, intercept, samples=2000):
    """The separation between the two flights across the time they share.

    This is the quantity the search minimised, so drawing it says what the angle
    bought rather than only where the two lines cross.
    """
    shared_end = min(primary.in_flight_time[-1], intercept.interceptor_time[-1])
    shared = np.linspace(0.0, shared_end, samples)
    primary_track = np.column_stack(
        [np.interp(shared, primary.in_flight_time, primary.in_flight[:, axis]) for axis in range(3)]
    )
    interceptor_track = np.column_stack(
        [np.interp(shared, intercept.interceptor_time, intercept.interceptor[:, axis]) for axis in range(3)]
    )
    return shared, np.linalg.norm(primary_track - interceptor_track, axis=1)


def describe_launch():
    """The conditions the figure was produced under, printed and drawn alike."""
    return "\n".join(
        [
            f"primary v0 = {PRIMARY_SPEED_MPS:.0f} m/s at {PRIMARY_ELEVATION_DEG:.0f} deg,   "
            f"interceptor v0 = {INTERCEPTOR_SPEED_MPS:.0f} m/s,   "
            f"launch delay = {LAUNCH_DELAY_S:.0f} s",
            describe_atmosphere([f"latitude = 0 deg   replay step = {REPLAY_MAX_STEP_S} s"]),
        ]
    )


def render(primary, intercept, output_path):
    """The chase from the side, and the separation the search drove down."""
    meet = position_at(primary.in_flight, primary.in_flight_time, intercept.miss_time)

    figure, (side, separation_axis) = plt.subplots(1, 2, figsize=(13.0, 5.5))

    # The primary's whole flight, with the part already covered before the
    # interceptor launched drawn faintly. Without it the blue arc appears out of
    # nowhere and the delay means nothing to a reader.
    side.plot(primary.full[:, 0], primary.full[:, 1], color="#2c6fbb", linewidth=1.0, alpha=0.35)
    side.plot(
        primary.in_flight[:, 0],
        primary.in_flight[:, 1],
        color="#2c6fbb",
        linewidth=1.8,
        label="primary, in flight",
    )
    side.plot(
        intercept.interceptor[:, 0],
        intercept.interceptor[:, 1],
        color="#c0392b",
        linewidth=1.8,
        label="interceptor",
    )
    side.plot([0.0], [0.0], marker="o", color="#333333", markersize=7, linestyle="None", label="launcher")
    side.plot([meet[0]], [meet[1]], marker="x", color="#e07b39", markersize=11, markeredgewidth=2.5)
    side.annotate(
        f"closest approach: {intercept.miss_distance:.1f} m at {intercept.miss_time:.1f} s",
        xy=(meet[0], meet[1]),
        xytext=(0.34, 0.34),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.9},
        fontsize=10,
        color="#333333",
    )
    side.set_xlabel("downrange (m)")
    side.set_ylabel("altitude (m)")
    side.set_title("The chase from the side", fontsize=12)
    side.legend(loc="upper left", fontsize=9, framealpha=0.95)
    side.grid(True, alpha=0.3)

    # Separation against time rather than a plan view: both flights stay on the
    # centreline in this scenario, so a plan view is two lines on top of each
    # other and says nothing.
    shared, separation = separation_over_time(primary, intercept)
    separation_axis.plot(shared, separation, color="#4f8f3f", linewidth=1.8)
    separation_axis.plot(
        [intercept.miss_time],
        [intercept.miss_distance],
        marker="x",
        color="#e07b39",
        markersize=11,
        markeredgewidth=2.5,
    )
    separation_axis.annotate(
        f"{intercept.miss_distance:.2f} m",
        xy=(intercept.miss_time, intercept.miss_distance),
        xytext=(0.62, 0.86),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.9},
        fontsize=10,
        color="#333333",
    )
    separation_axis.set_yscale("log")
    separation_axis.set_xlabel("time after the interceptor launches (s)")
    separation_axis.set_ylabel("separation (m)")
    separation_axis.set_title("What the search minimised", fontsize=12)
    separation_axis.grid(True, which="both", alpha=0.3)
    separation_axis.text(
        0.03,
        0.06,
        f"the two meet {meet[0]:.0f} m downrange, {meet[1]:.0f} m up\n"
        f"the primary has been flying {LAUNCH_DELAY_S + intercept.miss_time:.1f} s by then",
        transform=separation_axis.transAxes,
        fontsize=9,
        color="#444444",
        ha="left",
        va="bottom",
    )

    figure.suptitle(
        f"Interceptor at {intercept.angle:.1f} deg, launched {LAUNCH_DELAY_S:.0f} s late: "
        f"{intercept.miss_distance:.1f} m from a primary at {meet[1]:.0f} m",
        fontsize=13,
    )
    figure.text(0.5, 0.015, describe_launch(), ha="center", fontsize=8.5, color="#333333")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(output_path=OUTPUT_PATH):
    primary, intercept = solve(BallisticPhysics())
    render(primary, intercept, output_path)

    meet = position_at(primary.in_flight, primary.in_flight_time, intercept.miss_time)
    print("=== Intercept ===")
    print("Launch conditions")
    print(describe_launch())
    print(f"Interceptor angle: {intercept.angle:.2f} deg")
    print(f"Interceptor launch delay: {LAUNCH_DELAY_S:.2f} s")
    print(f"Miss distance: {intercept.miss_distance:.2f} m")
    print(f"Miss time: {intercept.miss_time:.3f} s")
    print(f"Primary range at intercept: {meet[0]:.2f} m")
    print(f"Primary altitude at intercept: {meet[1]:.2f} m")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    run(main, OUTPUT_PATH, "Launch an interceptor at a primary already in flight and draw the meeting")
