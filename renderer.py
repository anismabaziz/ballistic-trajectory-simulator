import matplotlib.pyplot as plt

def plot_trajectory(x, y, R, T, H):

    plt.plot(x, y)

    plt.xlabel("Horizontal position")
    plt.ylabel("Vertical position")
    plt.title("Projectile Trajectory")

    # Add text inside the plot
    plt.text(R * 0.7, max(y) * 0.8, f"Time of flight = {T:.2f} s")
    plt.text(R * 0.7, max(y) * 0.7, f"Range = {R:.2f} m")
    plt.text(R * 0.7, max(y) * 0.6, f"Max Height = {H:.2f} m")

    plt.show()