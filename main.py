"""The command line: one command that solves, one command that renders.

The project is a solver with a demo attached, and the interface says so.
`solve` searches for a launch angle that reaches a target and reports what it
found. `render` opens the real-time window, where the same searches run against
a moving target under the keyboard.

`solve` completes without a display, which is what `--headless` is for: it selects
the non-interactive matplotlib backend and closes the figure instead of showing
it. `render` has no such flag. A window is the whole point of it.
"""

import argparse
import functools
import sys
from dataclasses import dataclass

import matplotlib
import numpy as np

# Select the non-interactive backend before pyplot arrives through utils, or the
# call lands too late to have any effect. Reading sys.argv rather than the parsed
# arguments is deliberate: this has to happen at import time, before argparse runs.
if "--headless" in sys.argv:
    matplotlib.use("Agg")

from ballistics.physics import BallisticPhysics
from ballistics.searches import find_launch_angle, solve_moving_target_angle
from ballistics.targets import Target, check_collision
from utils import animate_trajectory, plot_trajectory_3d

DEFAULT_LAUNCH_SPEED_MPS = 300.0
DEFAULT_TARGET_X_M = 3500.0
DEFAULT_TARGET_RADIUS_M = 20.0
DEFAULT_RENDER_TARGET_X_M = 2800.0
DEFAULT_LAUNCH_ELEVATION_DEG = 35.0
DEFAULT_LAUNCH_AZIMUTH_DEG = 0.0


@dataclass(frozen=True)
class LaunchSolution:
    """A launch angle, the flight it flies, and what that flight achieved.

    The searches in `ballistics.searches` hand back dictionaries. This is the same
    information under names the caller has to remember, so the report and the plot
    read it rather than spelling string keys at each other.
    """

    angle: float
    hit: bool
    hit_index: int | None
    miss_index: int | None
    miss_distance: float
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    time: np.ndarray


def solve_launch_solution(simulator, launch_speed, target, apply_earth_curvature=False):
    """A launch angle that reaches the target, and what it actually achieved.

    A stationary target is solved by bracketing its range and refining, which is
    exact for a projectile that only has to cover ground. A target that moves is
    solved by scoring the miss distance over a sweep of angles, because there is
    no range to bracket: the target is somewhere else by the time the shot lands.

    A stationary target out of reach has no bracket either, so it falls back to the
    same sweep rather than raising. A target too far away is the most ordinary way
    to misuse a gun, and the sweep answers it with the closest shot found and how
    far short it landed, which is something a user can act on.

    The trajectory both searches integrate over carries the curvature flag, so the
    angle and the flight that gets reported come from one model. Curvature that
    only bent the drawn flight would report a miss distance the picture contradicted.
    """
    trajectory_func = functools.partial(
        simulator.trajectory_3d,
        apply_earth_curvature=apply_earth_curvature,
    )

    if target.vx != 0.0:
        return sweep_for_launch_solution(launch_speed, target, trajectory_func)

    try:
        angle = find_launch_angle(launch_speed, target.x, trajectory_func=trajectory_func)
    except ValueError:
        return sweep_for_launch_solution(launch_speed, target, trajectory_func)

    xs, ys, zs, t_array, _range, _flight_time, _peak_altitude = trajectory_func(
        launch_speed, angle, return_time=True
    )
    hit, hit_index, miss_distance, miss_index = check_collision(xs, ys, zs, target, t_array=t_array)
    return LaunchSolution(
        angle=angle,
        hit=hit,
        hit_index=hit_index,
        miss_index=miss_index,
        miss_distance=float(miss_distance),
        xs=xs,
        ys=ys,
        zs=zs,
        time=t_array,
    )


def sweep_for_launch_solution(launch_speed, target, trajectory_func):
    """The angle that comes closest to the target, found by scoring a sweep."""
    found = solve_moving_target_angle(launch_speed, target, trajectory_func=trajectory_func)
    return LaunchSolution(
        angle=found["angle"],
        hit=found["hit"],
        hit_index=found["hit_idx"],
        miss_index=found["miss_index"],
        miss_distance=found["miss_distance"],
        xs=found["trajectory"][0],
        ys=found["trajectory"][1],
        zs=found["trajectory"][2],
        time=found["time"],
    )


def report_solution(solution, launch_speed, target):
    motion = "stationary" if target.vx == 0.0 else f"moving at {target.vx:.2f} m/s"
    print("=== Launch solution ===")
    print(f"Launch speed: {launch_speed:.2f} m/s")
    print(f"Target: {motion} at {target.x:.2f} m, radius {target.radius:.2f} m")
    print(f"Launch angle: {solution.angle:.2f} degrees")
    print(f"Hit: {solution.hit}")
    print(f"Miss distance: {solution.miss_distance:.2f} m")


def run_solve(args):
    simulator = BallisticPhysics()
    target = Target(x=args.target_x, radius=args.target_radius, vx=args.target_velocity_x)
    solution = solve_launch_solution(
        simulator,
        args.launch_speed,
        target,
        apply_earth_curvature=args.enable_earth_curvature,
    )
    report_solution(solution, args.launch_speed, target)

    target_positions = target.positions_over_time(solution.time)

    if args.output_gif_path:
        animate_trajectory(
            solution.xs,
            solution.ys,
            save_gif_path=args.output_gif_path,
            fps=args.output_gif_fps,
            show_plot=not args.headless,
            target_positions=target_positions,
            target_radius=target.radius,
            hit=solution.hit,
            hit_idx=solution.hit_index,
            miss_distance=solution.miss_distance,
        )
        return

    plot_trajectory_3d(
        solution.xs,
        solution.ys,
        solution.zs,
        target_positions=target_positions,
        target_radius=target.radius,
        title="Launch solution",
        show_plot=not args.headless,
    )


def run_render(args):
    from sim.simulation import PygameBallisticSimulation

    simulator = BallisticPhysics()
    PygameBallisticSimulation(
        simulator=simulator,
        launch_speed=args.launch_speed,
        launch_elevation_deg=args.launch_elevation_deg,
        launch_azimuth_deg=args.launch_azimuth_deg,
        target_x=args.target_x,
        target_radius=args.target_radius,
        target_velocity_x=args.target_velocity_x,
    ).run()


def add_solve_arguments(parser):
    """The flags `solve` takes.

    A function rather than a parser so the same set of flags can be read on its
    own by the checks that compare the documentation against the interface,
    without reaching into argparse internals to do it.
    """
    parser.add_argument("--launch-speed", type=float, default=DEFAULT_LAUNCH_SPEED_MPS)
    parser.add_argument("--target-x", type=float, default=DEFAULT_TARGET_X_M)
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS_M)
    parser.add_argument("--target-velocity-x", type=float, default=0.0)
    parser.add_argument("--enable-earth-curvature", action="store_true")
    parser.add_argument("--output-gif-path", type=str, default=None)
    parser.add_argument("--output-gif-fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")


def add_render_arguments(parser):
    """The flags `render` takes. The window has no headless flag; it is the point."""
    parser.add_argument("--launch-speed", type=float, default=DEFAULT_LAUNCH_SPEED_MPS)
    parser.add_argument("--launch-elevation-deg", type=float, default=DEFAULT_LAUNCH_ELEVATION_DEG)
    parser.add_argument("--launch-azimuth-deg", type=float, default=DEFAULT_LAUNCH_AZIMUTH_DEG)
    parser.add_argument("--target-x", type=float, default=DEFAULT_RENDER_TARGET_X_M)
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS_M)
    parser.add_argument("--target-velocity-x", type=float, default=0.0)


def build_cli_parser():
    parser = argparse.ArgumentParser("Ballistic Trajectory Simulator")
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="solve a launch angle against a target and report it")
    add_solve_arguments(solve)

    render = commands.add_parser("render", help="run the real-time renderer")
    add_render_arguments(render)

    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.command == "render":
        run_render(args)
    else:
        run_solve(args)


if __name__ == "__main__":
    main()
