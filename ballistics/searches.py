import numpy as np
from scipy.optimize import brentq

from ballistics.targets import check_collision, miss_distance_between_trajectories


def find_launch_angle(v0, target_x, trajectory_func):
    """
    Use brentq to find the launch angle that hits target_x.
    Automatically scans for a bracket to avoid ValueError.
    """
    grid = np.linspace(1.0, 89.0, 177)
    diffs = []
    for angle in grid:
        _, _, _, R, _, _ = trajectory_func(v0, angle)
        diffs.append(R - target_x)

    angle_low = None
    angle_high = None
    for i in range(len(grid) - 1):
        d1, d2 = diffs[i], diffs[i + 1]
        if d1 == 0:
            return float(grid[i])
        if d1 * d2 < 0:
            angle_low = grid[i]
            angle_high = grid[i + 1]
            break

    if angle_low is None or angle_high is None:
        raise ValueError("Cannot find bracketing angles. Target might be out of range.")

    def range_error(angle_deg):
        _, _, _, R, _, _ = trajectory_func(v0, angle_deg)
        return R - target_x

    return brentq(range_error, angle_low, angle_high)


def solve_moving_target_angle(v0, target, trajectory_func):
    """
    Find launch angle that minimizes distance to a moving target.
    """
    def miss_distance_at(angle_deg):
        xs, ys, zs, t_arr, _, _, _ = trajectory_func(v0, angle_deg, return_time=True, max_step=0.1)
        _, _, miss_distance, _ = check_collision(xs, ys, zs, target, t_array=t_arr)
        return miss_distance

    coarse_angles = np.linspace(1.0, 89.0, 89)
    coarse_misses = [miss_distance_at(a) for a in coarse_angles]
    coarse_idx = int(np.argmin(coarse_misses))
    coarse_best = coarse_angles[coarse_idx]

    low = max(1.0, coarse_best - 1.0)
    high = min(89.0, coarse_best + 1.0)
    fine_angles = np.linspace(low, high, 101)
    fine_misses = [miss_distance_at(a) for a in fine_angles]
    best_angle = float(fine_angles[int(np.argmin(fine_misses))])

    xs, ys, zs, t_arr, _, _, _ = trajectory_func(v0, best_angle, return_time=True, max_step=0.05)
    hit, hit_idx, miss_distance, miss_index = check_collision(
        xs,
        ys,
        zs,
        target,
        t_array=t_arr,
    )
    return {
        "angle": best_angle,
        "hit": hit,
        "hit_idx": hit_idx,
        "miss_index": miss_index,
        "miss_distance": float(miss_distance),
        "trajectory": (xs, ys, zs),
        "time": t_arr,
    }


def solve_interceptor_angle(primary_traj, primary_time, interceptor_speed, trajectory_func):
    """
    Launch a second missile from origin and minimize distance to primary missile.
    """
    def objective(angle_deg):
        xs2, ys2, zs2, t2, _, _, _ = trajectory_func(
            interceptor_speed,
            angle_deg,
            return_time=True,
            max_step=0.1,
        )
        traj2 = np.column_stack((xs2, ys2, zs2))
        miss_distance, _, _ = miss_distance_between_trajectories(
            primary_traj, traj2, primary_time, t2
        )
        return miss_distance

    coarse_angles = np.linspace(1.0, 89.0, 89)
    coarse_misses = [objective(a) for a in coarse_angles]
    coarse_idx = int(np.argmin(coarse_misses))
    coarse_best = coarse_angles[coarse_idx]

    low = max(1.0, coarse_best - 1.0)
    high = min(89.0, coarse_best + 1.0)
    fine_angles = np.linspace(low, high, 101)
    fine_misses = [objective(a) for a in fine_angles]
    best_angle = float(fine_angles[int(np.argmin(fine_misses))])

    xs2, ys2, zs2, t2, _, _, _ = trajectory_func(interceptor_speed, best_angle, return_time=True, max_step=0.05)
    traj2 = np.column_stack((xs2, ys2, zs2))
    miss_distance, t_shared, idx = miss_distance_between_trajectories(
        primary_traj, traj2, primary_time, t2
    )
    return {
        "angle": best_angle,
        "miss_distance": float(miss_distance),
        "shared_time": float(t_shared),
        "shared_index": idx,
        "trajectory": (xs2, ys2, zs2),
        "time": t2,
    }