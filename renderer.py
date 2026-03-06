import matplotlib.pyplot as plt


def plot_trajectory(results: list):
    # get all built-in color in plt 
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, item in enumerate(results):
      x, y, R, T, H, angle = item
      plt.plot(x, y, label=f"{angle}°", color=colors[idx % len(colors)])

    plt.xlabel("Horizontal position (m)")
    plt.ylabel("Vertical position (m)")
    plt.title("Projectile Trajectories")
    plt.legend()
    plt.show()


def show_result_table(results: list):
  
  print(f"{'Angle (°)':<10}{'Range (m)':<12}{'Time of Flight (s)':<18}{'Max Height (m)':<15}")

  for item in results:
    _, _, R, T, H, angle = item
    print(f"{angle:<10}{R:<12.2f}{T:<18.2f}{H:<15.2f}")


def print_optimal_angle(results: list):
  
  _, _, R, _, _, angle = results[0]
  optimal_R = R
  optimal_angle = angle

  for item in results:
    _, _, R, _, _, angle = item

    if R > optimal_R:
      optimal_R = R
      optimal_angle = angle

  print(f"Optimal Angle: {optimal_angle:.2f}deg")