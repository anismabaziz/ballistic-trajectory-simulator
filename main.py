from physics import trajectory_3d
from targets import Target, check_collision
from utils import (
    plot_trajectory,
    find_launch_angle,
    solve_moving_target_angle,
    solve_interceptor_angle,
    plot_intercept_trajectories,
)


def run_stationary_target_phase(v0=300.0, target_x=3500.0, target_radius=10.0):
    target = Target(x=target_x, radius=target_radius)

    theta_hit = find_launch_angle(v0, target.x)
    xs, ys, zs, t_arr, _, _, _ = trajectory_3d(v0, theta_hit, return_time=True)
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


def run_moving_target_phase(v0=300.0, target_x=3000.0, target_radius=20.0, vx=40.0):
    moving_target = Target(x=target_x, radius=target_radius, vx=vx)
    result = solve_moving_target_angle(v0, moving_target)

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


def run_interceptor_phase(primary_traj, primary_time, interceptor_speed=320.0):
    result = solve_interceptor_angle(primary_traj, primary_time, interceptor_speed)
    interceptor_traj = result["trajectory"]

    print("\n=== Phase 5.5 + 5.6: Interceptor + closest approach metric ===")
    print(f"Interceptor launch angle: {result['angle']:.2f} degrees")
    print(f"Closest missile-to-missile distance: {result['closest_distance']:.2f} m")
    print(f"Time of closest approach: {result['shared_time']:.3f} s")

    plot_intercept_trajectories(primary_traj, interceptor_traj)


def main():
    _, stationary_traj, stationary_time = run_stationary_target_phase()
    run_moving_target_phase()
    run_interceptor_phase(stationary_traj, stationary_time)


if __name__ == "__main__":
    main()
