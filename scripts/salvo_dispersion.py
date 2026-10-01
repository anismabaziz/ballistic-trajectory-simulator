"""A salvo of rounds off one launcher, and the spread they cover on the ground.

Nine rounds leave at the same speed and the same elevation and differ only in
azimuth, so the only thing separating their impact points is cross-range. The
figure shows the arcs from the side, the footprints where they land, and the
width of the pattern they cover, which is the number a salvo is fired for.

Every round flies through the same acceleration model the solver and the
renderer use, so the dispersion in the picture is the dispersion the project
actually computes rather than a drawing of it. The search never sees this
scenario; nothing aims a salvo here, it is only spread.

Run it with:

    uv run python scripts/salvo_dispersion.py
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ballistics.physics import BallisticPhysics
from figure_common import describe_atmosphere, run

# The launch the figure reports. The speed and elevation are the ones the
# renderer defaults to, so the arcs are the flight a reader would see in the
# window; the azimuth span is the spread this scenario fires.
LAUNCH_SPEED_MPS = 300.0
LAUNCH_ELEVATION_DEG = 35.0
AZIMUTH_CENTRE_DEG = 0.0
AZIMUTH_SPAN_DEG = 30.0
ROUND_COUNT = 9

# The step the rounds are integrated at. Fine enough that the impact points are
# the integrator's answer rather than a coarse rendering of it.
MAX_STEP_S = 0.05

OUTPUT_PATH = "figures/salvo_dispersion.png"


@dataclass(frozen=True)
class Round:
    """One round's flight, and where it put itself on the ground."""

    azimuth: float
    # Ground distance under the round at each sample, which is what a side view
    # plots against altitude. Folding the two horizontal axes together is what
    # makes the arcs readable at all.
    ground_track: np.ndarray
    altitude: np.ndarray
    time: np.ndarray
    impact_range: float
    impact_cross_range: float
    flight_time: float


def azimuths():
    """The azimuths the salvo covers, centred on `AZIMUTH_CENTRE_DEG`."""
    half_span = AZIMUTH_SPAN_DEG * 0.5
    return np.linspace(AZIMUTH_CENTRE_DEG - half_span, AZIMUTH_CENTRE_DEG + half_span, ROUND_COUNT)


def fly_salvo(physics):
    """The flown rounds, one flight per azimuth."""
    rounds = []
    for azimuth in azimuths():
        xs, ys, zs, t_array, ground_range, flight_time, _peak = physics.trajectory_3d(
            LAUNCH_SPEED_MPS,
            LAUNCH_ELEVATION_DEG,
            azimuth_deg=float(azimuth),
            return_time=True,
            max_step=MAX_STEP_S,
        )
        rounds.append(
            Round(
                azimuth=float(azimuth),
                ground_track=np.hypot(xs, zs),
                altitude=ys,
                time=t_array,
                impact_range=float(ground_range),
                impact_cross_range=float(zs[-1]),
                flight_time=float(flight_time),
            )
        )
    return rounds


def cross_range_spread(rounds):
    """The width of the pattern on the ground, measured across the impact points."""
    impacts = np.array([shot.impact_cross_range for shot in rounds])
    return float(impacts.max() - impacts.min())


def downrange_spread(rounds):
    """How much the azimuth spread costs in range, which is the cost of a salvo."""
    ranges = np.array([shot.impact_range for shot in rounds])
    return float(ranges.max() - ranges.min())


def describe_launch():
    """The conditions the figure was produced under, printed and drawn alike."""
    return "\n".join(
        [
            f"v0 = {LAUNCH_SPEED_MPS:.0f} m/s   elevation = {LAUNCH_ELEVATION_DEG:.0f} deg   "
            f"azimuth = {AZIMUTH_CENTRE_DEG - AZIMUTH_SPAN_DEG / 2:.0f} to "
            f"{AZIMUTH_CENTRE_DEG + AZIMUTH_SPAN_DEG / 2:.0f} deg in {ROUND_COUNT} rounds",
            describe_atmosphere([f"latitude = 0 deg   step = {MAX_STEP_S} s"]),
        ]
    )


def render(rounds, output_path):
    """The side view of the arcs, the footprints they land in, and the pattern width."""
    figure, (side, footprint) = plt.subplots(1, 2, figsize=(13.0, 5.5))
    fired = azimuths()
    colour_map = matplotlib.colormaps["viridis"]
    # One scale across both panels, so a colour means the same azimuth in the
    # side view as it does in the footprint.
    colour_scale = matplotlib.colors.Normalize(vmin=float(fired.min()), vmax=float(fired.max()))

    for shot, azimuth in zip(rounds, fired):
        side.plot(
            shot.ground_track,
            shot.altitude,
            color=colour_map(colour_scale(azimuth)),
            alpha=0.85,
            linewidth=1.4,
        )
    # Headroom above the apex, so the note has somewhere to sit that nothing is
    # drawn through.
    peak = max(float(shot.altitude.max()) for shot in rounds)
    side.set_ylim(0.0, peak * 1.3)
    side.set_xlabel("ground track (m)")
    side.set_ylabel("altitude (m)")
    side.set_title("Every round flies the same profile", fontsize=12)
    side.grid(True, alpha=0.3)
    side.text(
        0.03,
        0.97,
        f"the arcs fall on top of each other: a {AZIMUTH_SPAN_DEG:.0f} deg azimuth fan moves the\n"
        "rounds sideways without changing the flight, so the only cost is range",
        transform=side.transAxes,
        fontsize=9,
        color="#444444",
        ha="left",
        va="top",
    )

    impact_ranges = [shot.impact_range for shot in rounds]
    impact_cross_ranges = [shot.impact_cross_range for shot in rounds]
    spread = cross_range_spread(rounds)

    footprint.scatter(
        impact_cross_ranges,
        impact_ranges,
        c=fired,
        cmap=colour_map,
        norm=colour_scale,
        s=55,
        zorder=3,
    )
    footprint.plot(impact_cross_ranges, impact_ranges, linewidth=0.8, color="#666666", linestyle="--")
    footprint.axvline(0.0, color="#333333", linewidth=1.0, linestyle=":")
    footprint.annotate(
        f"pattern {spread:.0f} m wide",
        xy=(0.0, float(np.mean(impact_ranges))),
        xytext=(0.04, 0.08),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.9},
        fontsize=10,
        color="#333333",
    )
    footprint.set_xlabel("cross-range (m)")
    footprint.set_ylabel("downrange (m)")
    footprint.set_title("Where the rounds land", fontsize=12)
    footprint.grid(True, alpha=0.3)
    colour_bar = figure.colorbar(
        footprint.collections[0],
        ax=footprint,
        fraction=0.046,
        pad=0.03,
    )
    colour_bar.set_label("launch azimuth (deg)")

    figure.suptitle(
        f"{ROUND_COUNT} rounds, {AZIMUTH_SPAN_DEG:.0f} deg of azimuth: "
        f"{spread:.0f} m across, {downrange_spread(rounds):.0f} m of range spent",
        fontsize=13,
    )
    figure.text(0.5, 0.015, describe_launch(), ha="center", fontsize=8.5, color="#333333")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(output_path=OUTPUT_PATH):
    rounds = fly_salvo(BallisticPhysics())
    render(rounds, output_path)

    print("=== Salvo dispersion ===")
    print("Launch conditions")
    print(describe_launch())
    print(f"Rounds: {len(rounds)}")
    print(f"Impact cross-range spread: {cross_range_spread(rounds):.2f} m")
    print(f"Impact downrange spread: {downrange_spread(rounds):.2f} m")
    for shot in rounds:
        print(
            f"  azimuth={shot.azimuth:+.1f} deg  range={shot.impact_range:.1f} m  "
            f"cross-range={shot.impact_cross_range:+.1f} m  flight={shot.flight_time:.2f} s"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    run(main, OUTPUT_PATH, "Fly a salvo across a range of azimuths and draw the pattern")
