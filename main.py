from physics import trajectory
from renderer import plot_trajectory


x, y, R, T, H = trajectory(100, 45)

plot_trajectory(x, y, R, T, H)