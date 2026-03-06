import argparse
import numpy as np
from physics import BallisticPhysics
from targets import Target, check_collision
from utils import (
    animate_trajectory,
    plot_trajectory,
    find_launch_angle,
    solve_moving_target_angle,
    solve_interceptor_angle,
    plot_intercept_trajectories,
)


def run_stationary_target_phase(simulator, v0=300.0, target_x=3500.0, target_radius=10.0):
    target = Target(x=target_x, radius=target_radius)

    theta_hit = find_launch_angle(v0, target.x, trajectory_func=simulator.trajectory_3d)
    xs, ys, zs, t_arr, _, _, _ = simulator.trajectory_3d(v0, theta_hit, return_time=True)
    hit, hit_idx, closest_distance, closest_idx = check_collision(xs, ys, zs, target, t_array=t_arr)

    target_positions = target.positions_over_time(t_arr)
    print("=== Phase 5.1 + 5.2: Stationary target + hit/miss indicator ===")
    print(f"Launch angle: {theta_hit:.2f} degrees")
    print(f"Hit: {hit}")
    print(f"Closest distance: {closest_distance:.2f} m")

    plot_trajectory(
        xs,
        ys,
        target,
        hit=hit,
        hit_idx=hit_idx,
        closest_idx=closest_idx,
        target_positions=target_positions,
        closest_distance=closest_distance,
        title="Stationary target collision check",
    )

    return theta_hit, (xs, ys, zs), t_arr


def run_moving_target_phase(simulator, v0=300.0, target_x=3000.0, target_radius=20.0, vx=40.0):
    moving_target = Target(x=target_x, radius=target_radius, vx=vx)
    result = solve_moving_target_angle(v0, moving_target, trajectory_func=simulator.trajectory_3d)

    xs, ys, zs = result["trajectory"]
    t_arr = result["time"]
    target_positions = moving_target.positions_over_time(t_arr)

    print("\n=== Phase 5.3 + 5.4: Angle solver for moving target ===")
    print(f"Solved launch angle: {result['angle']:.2f} degrees")
    print(f"Hit: {result['hit']}")
    print(f"Closest distance: {result['closest_distance']:.2f} m")

    plot_trajectory(
        xs,
        ys,
        moving_target,
        hit=result["hit"],
        hit_idx=result["hit_idx"],
        closest_idx=result["closest_idx"],
        target_positions=target_positions,
        closest_distance=result["closest_distance"],
        title="Moving target interception",
    )

    return result


def run_interceptor_phase(simulator, primary_traj, primary_time, interceptor_speed=320.0):
    result = solve_interceptor_angle(
        primary_traj,
        primary_time,
        interceptor_speed,
        trajectory_func=simulator.trajectory_3d,
    )
    interceptor_traj = result["trajectory"]

    print("\n=== Phase 5.5 + 5.6: Interceptor + closest approach metric ===")
    print(f"Interceptor launch angle: {result['angle']:.2f} degrees")
    print(f"Closest missile-to-missile distance: {result['closest_distance']:.2f} m")
    print(f"Time of closest approach: {result['shared_time']:.3f} s")

    plot_intercept_trajectories(primary_traj, interceptor_traj)


def run_animation_phase(
    simulator,
    v0=300.0,
    angle_deg=35.0,
    target_x=None,
    target_radius=20.0,
    target_vx=0.0,
    interval_ms=30,
    save_gif_path=None,
    gif_fps=30,
    show_plot=True,
):
    xs, ys, _, t_arr, _, _, _ = simulator.trajectory_3d(v0, angle_deg, return_time=True, max_step=0.05)
    target_positions = None
    hit = None
    hit_idx = None
    closest_distance = None

    if target_x is not None:
        target = Target(x=target_x, radius=target_radius, vx=target_vx)
        target_positions = target.positions_over_time(t_arr)
        hit, hit_idx, closest_distance, _ = check_collision(
            xs,
            ys,
            np.zeros_like(xs),
            target,
            t_array=t_arr,
        )

        print("=== Animation target state ===")
        print(f"Target mode: {'moving' if target_vx != 0 else 'stationary'}")
        print(f"Hit: {hit}")
        print(f"Closest distance: {closest_distance:.2f} m")

    animate_trajectory(
        xs,
        ys,
        interval_ms=interval_ms,
        save_gif_path=save_gif_path,
        fps=gif_fps,
        show_plot=show_plot,
        target_positions=target_positions,
        target_radius=target_radius if target_positions is not None else None,
        hit=hit,
        hit_idx=hit_idx,
        closest_distance=closest_distance,
    )


def build_cli_parser():
    parser = argparse.ArgumentParser("Ballistic Trajectory Simulator")
    parser.add_argument("--mode", choices=["phase5", "animate"], default="phase5")
    parser.add_argument("--velocity", type=float, default=300.0)
    parser.add_argument("--angle", type=float, default=35.0)
    parser.add_argument("--interval-ms", type=int, default=30)
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-radius", type=float, default=20.0)
    parser.add_argument("--target-vx", type=float, default=0.0)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_cli_parser().parse_args()
    simulator = BallisticPhysics()

    if args.mode == "animate":
        run_animation_phase(
            simulator,
            v0=args.velocity,
            angle_deg=args.angle,
            target_x=args.target_x,
            target_radius=args.target_radius,
            target_vx=args.target_vx,
            interval_ms=args.interval_ms,
            save_gif_path=args.save_gif,
            gif_fps=args.gif_fps,
            show_plot=not args.no_show,
        )
        return

    _, stationary_traj, stationary_time = run_stationary_target_phase(simulator, v0=args.velocity)
    run_moving_target_phase(simulator, v0=args.velocity)
    run_interceptor_phase(simulator, stationary_traj, stationary_time)


if __name__ == "__main__":
    main()
