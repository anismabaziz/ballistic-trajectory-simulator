import argparse
import numpy as np
from physics import BallisticPhysics
from sim.simulation import PygameBallisticSimulation
from targets import Target, check_collision
from utils import (
    animate_trajectory,
    plot_trajectory,
    plot_trajectory_3d,
    plot_salvo_dispersion_3d,
    find_launch_angle,
    solve_moving_target_angle,
    solve_interceptor_angle,
    plot_intercept_trajectories,
)


def run_stationary_target_scenario(simulator, v0=300.0, target_x=3500.0, target_radius=10.0):
    target = Target(x=target_x, radius=target_radius)

    theta_hit = find_launch_angle(v0, target.x, trajectory_func=simulator.trajectory_3d)
    xs, ys, zs, t_arr, _, _, _ = simulator.trajectory_3d(v0, theta_hit, return_time=True)
    hit, hit_idx, closest_distance, closest_idx = check_collision(xs, ys, zs, target, t_array=t_arr)

    target_positions = target.positions_over_time(t_arr)
    print("=== Stationary target result ===")
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


def run_moving_target_scenario(simulator, v0=300.0, target_x=3000.0, target_radius=20.0, vx=40.0):
    moving_target = Target(x=target_x, radius=target_radius, vx=vx)
    result = solve_moving_target_angle(v0, moving_target, trajectory_func=simulator.trajectory_3d)

    xs, ys, zs = result["trajectory"]
    t_arr = result["time"]
    target_positions = moving_target.positions_over_time(t_arr)

    print("\n=== Moving target result ===")
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


def run_interceptor_scenario(simulator, primary_traj, primary_time, interceptor_speed=320.0):
    result = solve_interceptor_angle(
        primary_traj,
        primary_time,
        interceptor_speed,
        trajectory_func=simulator.trajectory_3d,
    )
    interceptor_traj = result["trajectory"]

    print("\n=== Interceptor result ===")
    print(f"Interceptor launch angle: {result['angle']:.2f} degrees")
    print(f"Closest missile-to-missile distance: {result['closest_distance']:.2f} m")
    print(f"Time of closest approach: {result['shared_time']:.3f} s")

    plot_intercept_trajectories(primary_traj, interceptor_traj)


def run_real_time_animation(
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
    parser.add_argument(
        "--mode",
        choices=["target-intercept", "real-time-animation", "three-d-simulation", "interactive-simulator"],
        default="target-intercept",
    )
    parser.add_argument("--launch-speed", type=float, default=300.0)
    parser.add_argument("--launch-elevation-deg", type=float, default=35.0)
    parser.add_argument("--frame-interval-ms", type=int, default=30)
    parser.add_argument("--output-gif-path", type=str, default=None)
    parser.add_argument("--output-gif-fps", type=int, default=30)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-radius", type=float, default=20.0)
    parser.add_argument("--target-velocity-x", type=float, default=0.0)
    parser.add_argument("--launch-azimuth-deg", type=float, default=0.0)
    parser.add_argument("--enable-earth-curvature", action="store_true")
    parser.add_argument("--enable-salvo", action="store_true")
    parser.add_argument("--salvo-missile-count", type=int, default=9)
    parser.add_argument("--salvo-azimuth-span-deg", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true")
    return parser


def run_interactive_simulator(
    simulator,
    launch_speed,
    launch_elevation_deg,
    launch_azimuth_deg,
    target_x,
    target_radius,
    target_velocity_x,
):
    target_x_value = float(target_x) if target_x is not None else 2800.0
    ui = PygameBallisticSimulation(
        simulator=simulator,
        launch_speed=launch_speed,
        launch_elevation_deg=launch_elevation_deg,
        launch_azimuth_deg=launch_azimuth_deg,
        target_x=target_x_value,
        target_radius=target_radius,
        target_velocity_x=target_velocity_x,
    )
    ui.run()


def run_three_d_simulation(
    simulator,
    v0=300.0,
    angle_deg=35.0,
    azimuth_deg=0.0,
    target_x=None,
    target_radius=20.0,
    target_vx=0.0,
    apply_earth_curvature=False,
    salvo=False,
    salvo_count=9,
    salvo_span=30.0,
):
    if salvo:
        half_span = float(salvo_span) * 0.5
        azimuth_values = np.linspace(azimuth_deg - half_span, azimuth_deg + half_span, int(salvo_count))
        plot_salvo_dispersion_3d(
            simulator,
            v0,
            angle_deg,
            azimuth_values,
            apply_earth_curvature=apply_earth_curvature,
        )
        return

    xs, ys, zs, t_arr, _, _, _ = simulator.trajectory_3d(
        v0,
        angle_deg,
        azimuth_deg=azimuth_deg,
        return_time=True,
        apply_earth_curvature=apply_earth_curvature,
    )

    target_positions = None
    if target_x is not None:
        target = Target(x=target_x, radius=target_radius, vx=target_vx)
        target_positions = target.positions_over_time(t_arr)
        hit, _, closest_distance, _ = check_collision(xs, ys, zs, target, t_array=t_arr)
        print("=== 3D target state ===")
        print(f"Target mode: {'moving' if target_vx != 0 else 'stationary'}")
        print(f"Hit: {hit}")
        print(f"Closest distance: {closest_distance:.2f} m")

    plot_trajectory_3d(
        xs,
        ys,
        zs,
        target_positions=target_positions,
        target_radius=target_radius if target_positions is not None else None,
        title="3D Simulation",
    )


def main():
    args = build_cli_parser().parse_args()
    simulator = BallisticPhysics()

    if args.mode == "real-time-animation":
        run_real_time_animation(
            simulator,
            v0=args.launch_speed,
            angle_deg=args.launch_elevation_deg,
            target_x=args.target_x,
            target_radius=args.target_radius,
            target_vx=args.target_velocity_x,
            interval_ms=args.frame_interval_ms,
            save_gif_path=args.output_gif_path,
            gif_fps=args.output_gif_fps,
            show_plot=not args.headless,
        )
        return

    if args.mode == "three-d-simulation":
        run_three_d_simulation(
            simulator,
            v0=args.launch_speed,
            angle_deg=args.launch_elevation_deg,
            azimuth_deg=args.launch_azimuth_deg,
            target_x=args.target_x,
            target_radius=args.target_radius,
            target_vx=args.target_velocity_x,
            apply_earth_curvature=args.enable_earth_curvature,
            salvo=args.enable_salvo,
            salvo_count=args.salvo_missile_count,
            salvo_span=args.salvo_azimuth_span_deg,
        )
        return

    if args.mode == "interactive-simulator":
        run_interactive_simulator(
            simulator,
            launch_speed=args.launch_speed,
            launch_elevation_deg=args.launch_elevation_deg,
            launch_azimuth_deg=args.launch_azimuth_deg,
            target_x=args.target_x,
            target_radius=args.target_radius,
            target_velocity_x=args.target_velocity_x,
        )
        return

    _, stationary_traj, stationary_time = run_stationary_target_scenario(simulator, v0=args.launch_speed)
    run_moving_target_scenario(simulator, v0=args.launch_speed)
    run_interceptor_scenario(simulator, stationary_traj, stationary_time)


if __name__ == "__main__":
    main()
