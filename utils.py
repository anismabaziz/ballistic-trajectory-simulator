import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from scipy.optimize import brentq
from physics import BallisticPhysics
from targets import check_collision, closest_approach_between_trajectories


DEFAULT_SIMULATOR = BallisticPhysics()


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


def find_launch_angle(v0, target_x, trajectory_func=None):
    """
    Use brentq to find the launch angle that hits target_x.
    Automatically scans for a bracket to avoid ValueError.
    """
    if trajectory_func is None:
        trajectory_func = DEFAULT_SIMULATOR.trajectory_3d

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


def solve_moving_target_angle(v0, target, trajectory_func=None):
    """
    Find launch angle that minimizes distance to a moving target.
    """
    if trajectory_func is None:
        trajectory_func = DEFAULT_SIMULATOR.trajectory_3d

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


def solve_interceptor_angle(primary_traj, primary_time, interceptor_speed, trajectory_func=None):
    """
    Launch a second missile from origin and minimize distance to primary missile.
    """
    if trajectory_func is None:
        trajectory_func = DEFAULT_SIMULATOR.trajectory_3d

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


def animate_trajectory(
    xs,
    ys,
    interval_ms=30,
    save_gif_path=None,
    fps=30,
    show_plot=True,
    target_positions=None,
    target_radius=None,
    hit=None,
    hit_idx=None,
    closest_distance=None,
):
    """
    Animate a precomputed missile trajectory using matplotlib FuncAnimation.

    interval_ms controls playback speed (milliseconds per frame).
    save_gif_path optionally saves animation as a GIF with pillow writer.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x_pad = max((np.max(xs) - np.min(xs)) * 0.05, 1.0)
    y_pad = max((np.max(ys) - np.min(ys)) * 0.1, 1.0)
    ax.set_xlim(np.min(xs) - x_pad, np.max(xs) + x_pad)
    ax.set_ylim(min(0.0, np.min(ys) - y_pad), np.max(ys) + y_pad)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Phase 6 - Real-Time Missile Animation")
    ax.grid(True)

    trail_line, = ax.plot([], [], color="tab:blue", lw=2, label="Trail")
    missile_point, = ax.plot([], [], "o", color="crimson", markersize=8, label="Missile")

    target_path_line = None
    target_point = None
    target_circle = None
    if target_positions is not None:
        tx = target_positions[:, 0]
        ty = target_positions[:, 1]
        target_path_line, = ax.plot(tx, ty, "--", color="gray", alpha=0.8, label="Target path")
        target_point, = ax.plot([], [], "o", color="green", markersize=7, label="Target")
        if target_radius is not None:
            target_circle = mpatches.Circle((tx[0], ty[0]), target_radius, fill=False, color="green", alpha=0.8)
            ax.add_patch(target_circle)

    status_text = None
    if hit is not None:
        status_label = "HIT" if hit else "MISS"
        status_color = "green" if hit else "red"
        subtitle = f"{status_label}"
        if closest_distance is not None:
            subtitle += f" | Closest distance: {closest_distance:.2f} m"
        status_text = ax.text(
            0.02,
            0.98,
            subtitle,
            transform=ax.transAxes,
            verticalalignment="top",
            color=status_color,
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": status_color},
        )

    hit_marker, = ax.plot([], [], marker="o", color="limegreen", markersize=10, linestyle="None", label="Hit")

    ax.legend(loc="upper right")

    def init():
        trail_line.set_data([], [])
        missile_point.set_data([], [])
        artists = [trail_line, missile_point]
        if target_point is not None:
            target_point.set_data([], [])
            artists.append(target_point)
        if target_path_line is not None:
            artists.append(target_path_line)
        hit_marker.set_data([], [])
        artists.append(hit_marker)
        return tuple(artists)

    def update(i):
        trail_line.set_data(xs[: i + 1], ys[: i + 1])
        missile_point.set_data([xs[i]], [ys[i]])
        artists = [trail_line, missile_point]
        if target_positions is not None and target_point is not None:
            tx_i = target_positions[i, 0]
            ty_i = target_positions[i, 1]
            target_point.set_data([tx_i], [ty_i])
            artists.append(target_point)
            if target_circle is not None:
                target_circle.center = (tx_i, ty_i)
        if target_path_line is not None:
            artists.append(target_path_line)
        if hit and hit_idx is not None and i >= hit_idx:
            hit_marker.set_data([xs[hit_idx]], [ys[hit_idx]])
        else:
            hit_marker.set_data([], [])
        artists.append(hit_marker)
        return tuple(artists)

    use_blit = target_circle is None and status_text is None

    anim = FuncAnimation(
        fig,
        update,
        frames=len(xs),
        init_func=init,
        interval=interval_ms,
        blit=use_blit,
        repeat=False,
    )

    if save_gif_path:
        anim.save(save_gif_path, writer="pillow", fps=fps)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return anim
