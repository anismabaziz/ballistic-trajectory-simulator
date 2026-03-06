from physics import trajectory
from renderer import plot_trajectory, show_result_table, print_optimal_angle
from config import G, DEFAULT_VELOCITY, DEFAULT_ANGLES
import argparse


parser = argparse.ArgumentParser("Ballistic Trajectory Simulator")
parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY, help="Initial missile velocity (m/s)")
parser.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES, help="List of launch angles")

args = parser.parse_args()

v = args.velocity
angles = args.angles

result = []
for angle in angles:
  x, y, R, T, H = trajectory(v, angle)
  result.append((x, y, R, T, H, angle))

show_result_table(result)
print_optimal_angle(result)
plot_trajectory(result)
