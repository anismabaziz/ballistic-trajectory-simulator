from physics import trajectory_3d
from renderer import plot_trajectory, show_result_table, print_optimal_angle
from config import G, DEFAULT_VELOCITY, DEFAULT_ANGLES, ALT_LEVELS, WIND_X, WIND_Y
import argparse
import numpy as np

# Command-line parser
parser = argparse.ArgumentParser("Ballistic Trajectory Simulator")
parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY, help="Initial missile velocity (m/s)")
parser.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES, help="List of launch angles")
args = parser.parse_args()

v = args.velocity
angles = args.angles

# Wind & altitude config (from config by default)
alt_levels = np.array(ALT_LEVELS)
wind_x_vals = np.array(WIND_X)
wind_y_vals = np.array(WIND_Y)
latitude = 0  # optional for Coriolis effect

# Run simulation for all launch angles
result = []
for angle in angles:
    xs, ys, zs, R, T, H = trajectory_3d(
        v0=v,
        angle_deg=angle,
        alt_levels=alt_levels,
        wind_x_vals=wind_x_vals,
        wind_y_vals=wind_y_vals,
        latitude=latitude
    )
    # Store full 3D trajectory
    result.append((xs, ys, zs, R, T, H, angle))

# Display results
show_result_table(result)
print_optimal_angle(result)

# Plot 2D trajectory (x vs y) with wind arrows
plot_trajectory(result)