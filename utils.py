import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy.optimize import brentq
from physics import trajectory_3d
from targets import check_collision, closest_approach_between_trajectories


def plot_trajectory(
    xs,
    ys,
    target,
    hit=False,
    hit_idx=None,
    closest_idx=None,
    target_positions=None,
    closest_distance=None,
    title="Missile Trajectory",
):
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, label="Missile Path")

    if target_positions is None:
        target_positions = np.column_stack((
            np.full_like(xs, target.x, dtype=float),
            np.full_like(ys, target.y, dtype=float),
            np.zeros_like(xs, dtype=float),
        ))

    tx = target_positions[:, 0]
    ty = target_positions[:, 1]
    plt.plot(tx, ty, linestyle="--", color="gray", label="Target Path")

    if closest_idx is not None:
        plt.scatter(
            xs[closest_idx],
            ys[closest_idx],
            color="orange",
            s=60,
            label="Closest approach",
            zorder=5,
        )

    if hit:
        hit_index = hit_idx if hit_idx is not None else int(np.argmin(np.abs(xs - tx)))
        plt.scatter(xs[hit_index], ys[hit_index], color='green', s=100, label='Hit')
        circle = mpatches.Circle(
            (float(tx[hit_index]), float(ty[hit_index])),
            target.radius,
            color='green',
            fill=False,
        )
        plt.gca().add_patch(circle)
    else:
        miss_idx = closest_idx if closest_idx is not None else -1
        plt.scatter(tx[miss_idx], ty[miss_idx], color='red', marker='x', s=100, label='Miss')

    if closest_distance is not None:
        plt.text(
            0.02,
            0.98,
            f"Closest distance: {closest_distance:.2f} m",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def find_launch_angle(v0, target_x, trajectory_func=trajectory_3d):
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


def solve_moving_target_angle(v0, target, trajectory_func=trajectory_3d):
    """
    Find launch angle that minimizes distance to a moving target.
    """
    def miss_distance(angle_deg):
        xs, ys, zs, t_arr, _, _, _ = trajectory_func(v0, angle_deg, return_time=True, max_step=0.1)
        _, _, closest_distance, _ = check_collision(xs, ys, zs, target, t_array=t_arr)
        return closest_distance

    coarse_angles = np.linspace(1.0, 89.0, 89)
    coarse_misses = [miss_distance(a) for a in coarse_angles]
    coarse_idx = int(np.argmin(coarse_misses))
    coarse_best = coarse_angles[coarse_idx]

    low = max(1.0, coarse_best - 1.0)
    high = min(89.0, coarse_best + 1.0)
    fine_angles = np.linspace(low, high, 101)
    fine_misses = [miss_distance(a) for a in fine_angles]
    best_angle = float(fine_angles[int(np.argmin(fine_misses))])

    xs, ys, zs, t_arr, _, _, _ = trajectory_func(v0, best_angle, return_time=True, max_step=0.05)
    hit, hit_idx, closest_distance, closest_idx = check_collision(
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
        "closest_idx": closest_idx,
        "closest_distance": float(closest_distance),
        "trajectory": (xs, ys, zs),
        "time": t_arr,
    }


def solve_interceptor_angle(primary_traj, primary_time, interceptor_speed, trajectory_func=trajectory_3d):
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
        dmin, _, _ = closest_approach_between_trajectories(primary_traj, traj2, primary_time, t2)
        return dmin

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
    dmin, t_shared, idx = closest_approach_between_trajectories(primary_traj, traj2, primary_time, t2)
    return {
        "angle": best_angle,
        "closest_distance": float(dmin),
        "shared_time": float(t_shared),
        "shared_index": idx,
        "trajectory": (xs2, ys2, zs2),
        "time": t2,
    }


def plot_intercept_trajectories(primary_traj, interceptor_traj, title="Missile Intercept Scenario"):
    plt.figure(figsize=(10, 5))
    plt.plot(primary_traj[:, 0], primary_traj[:, 1], label="Primary missile")
    plt.plot(interceptor_traj[:, 0], interceptor_traj[:, 1], label="Interceptor missile")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
