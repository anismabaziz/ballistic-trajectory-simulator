import matplotlib.pyplot as plt
import numpy as np
from config import ALT_LEVELS, WIND_X, WIND_Y

def plot_trajectory(results: list, show_wind=True):
    """
    Plot 2D projectile trajectories (x vs y) with optional wind arrows.
    
    Each item in results: (xs, ys, zs, R, T, H, angle)
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, item in enumerate(results):
        xs, ys, zs, R, T, H, angle = item

        # Plot trajectory
        plt.plot(xs, ys, label=f"{angle}°", color=colors[idx % len(colors)])

        if show_wind:
            # Number of arrows along trajectory
            n_arrows = 10
            arrow_indices = np.linspace(0, len(xs)-1, n_arrows, dtype=int)
            arrow_x = xs[arrow_indices]
            arrow_y = ys[arrow_indices]

            # interpolate wind at current altitudes
            wind_u = np.interp(arrow_y, ALT_LEVELS, WIND_X)
            wind_v = np.interp(arrow_y, ALT_LEVELS, WIND_Y)

            # scale arrows relative to trajectory span
            dx_span = xs[-1] - xs[0]
            dy_span = max(ys) - min(ys)
            # fraction of the trajectory span to scale arrows
            arrow_scale = 0.05
            wind_u_scaled = wind_u * dx_span * arrow_scale / np.maximum(np.abs(wind_u), 1)
            wind_v_scaled = wind_v * dy_span * arrow_scale / np.maximum(np.abs(wind_v), 1)

            # draw arrows
            plt.quiver(
                arrow_x, arrow_y, wind_u_scaled, wind_v_scaled,
                color='blue', alpha=0.8, angles='xy', scale_units='xy', scale=1, width=0.005
            )

    plt.xlabel("Horizontal position (m)")
    plt.ylabel("Vertical position (m)")
    plt.title("Projectile Trajectories with Wind")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_result_table(results: list):
    """Prints a table of angles, range, time of flight, and max height."""
    print(f"{'Angle (°)':<10}{'Range (m)':<12}{'Time of Flight (s)':<22}{'Max Height (m)':<15}")
    for item in results:
        _, _, _, R, T, H, angle = item
        print(f"{angle:<10}{R:<12.2f}{T:<22.2f}{H:<15.2f}")


def print_optimal_angle(results: list):
    """Prints the launch angle that gives the maximum horizontal range."""
    optimal_item = max(results, key=lambda x: x[3])  # x[3] is R
    optimal_angle = optimal_item[6]                  # angle
    print(f"Optimal Angle: {optimal_angle:.2f}°")