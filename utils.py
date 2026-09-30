import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

def show_or_close(fig, show_plot):
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
def plot_trajectory(
    xs,
    ys,
    target,
    hit=False,
    hit_idx=None,
    miss_index=None,
    target_positions=None,
    miss_distance=None,
    title="Missile Trajectory",
    show_plot=True,
):
    fig = plt.figure(figsize=(10, 5))
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

    if miss_index is not None:
        plt.scatter(
            xs[miss_index],
            ys[miss_index],
            color="orange",
            s=60,
            label="Nearest pass",
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
        miss_idx = miss_index if miss_index is not None else -1
        plt.scatter(tx[miss_idx], ty[miss_idx], color='red', marker='x', s=100, label='Miss')

    if miss_distance is not None:
        plt.text(
            0.02,
            0.98,
            f"Miss distance: {miss_distance:.2f} m",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    show_or_close(fig, show_plot)
def plot_intercept_trajectories(
    primary_traj,
    interceptor_traj,
    title="Missile Intercept Scenario",
    show_plot=True,
):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(primary_traj[:, 0], primary_traj[:, 1], label="Primary missile")
    plt.plot(interceptor_traj[:, 0], interceptor_traj[:, 1], label="Interceptor missile")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    show_or_close(fig, show_plot)


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
    miss_distance=None,
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
        if miss_distance is not None:
            subtitle += f" | Miss distance: {miss_distance:.2f} m"
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

    show_or_close(fig, show_plot)

    return anim


def plot_trajectory_3d(
    xs,
    ys,
    zs,
    target_positions=None,
    target_radius=None,
    title="Phase 7 - 3D Trajectory",
    elev=25,
    azim=-60,
    show_plot=True,
):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xs, zs, ys, color="tab:blue", lw=2, label="Missile")

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    x_span = max(x_max - x_min, 1.0)
    z_span = max(z_max - z_min, 1.0)
    x_pad = 0.05 * x_span
    z_pad = 0.05 * z_span

    gx = np.linspace(x_min - x_pad, x_max + x_pad, 20)
    gz = np.linspace(z_min - z_pad, z_max + z_pad, 20)
    GX, GZ = np.meshgrid(gx, gz)
    GY = np.zeros_like(GX)
    ax.plot_surface(GX, GZ, GY, alpha=0.2, color="gray", linewidth=0)

    if target_positions is not None:
        tx = target_positions[:, 0]
        ty = target_positions[:, 1]
        tz = target_positions[:, 2]
        ax.plot(tx, tz, ty, linestyle="--", color="green", label="Target path")
        ax.scatter([tx[-1]], [tz[-1]], [ty[-1]], color="green", s=40, label="Target")

        if target_radius is not None:
            theta = np.linspace(0, 2 * np.pi, 80)
            cx = tx[-1] + target_radius * np.cos(theta)
            cz = tz[-1] + target_radius * np.sin(theta)
            cy = np.full_like(theta, ty[-1])
            ax.plot(cx, cz, cy, color="green", alpha=0.8)

    ax.set_xlabel("X (range)")
    ax.set_ylabel("Z (cross-range)")
    ax.set_zlabel("Y (altitude)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    plt.tight_layout()
    show_or_close(fig, show_plot)


def plot_salvo_dispersion_3d(
    simulator,
    v0,
    angle_deg,
    azimuth_values,
    max_step=0.05,
    apply_earth_curvature=False,
    show_plot=True,
):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    impact_x = []
    impact_z = []

    for az in azimuth_values:
        xs, ys, zs, _, _, _ = simulator.trajectory_3d(
            v0,
            angle_deg,
            azimuth_deg=float(az),
            max_step=max_step,
            apply_earth_curvature=apply_earth_curvature,
        )
        ax.plot(xs, zs, ys, alpha=0.8)
        impact_x.append(xs[-1])
        impact_z.append(zs[-1])

    ax.scatter(impact_x, impact_z, np.zeros_like(impact_x), color="red", s=30, label="Impact points")
    ax.set_xlabel("X (range)")
    ax.set_ylabel("Z (cross-range)")
    ax.set_zlabel("Y (altitude)")
    ax.set_title("Phase 7.6 - Multi-salvo dispersion")
    ax.view_init(elev=22, azim=-65)
    ax.legend()
    plt.tight_layout()
    show_or_close(fig, show_plot)
