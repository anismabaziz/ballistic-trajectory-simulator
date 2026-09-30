import numpy as np

class Target:
    def __init__(self, x: float, y: float = 0.0, z: float = 0.0,
                 radius: float = 10.0, vx: float = 0.0,
                 vy: float = 0.0, vz: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.vx = vx  # velocity for moving targets
        self.vy = vy
        self.vz = vz

    def position_at(self, t):
        return self.x + self.vx * t, self.y + self.vy * t, self.z + self.vz * t

    def positions_over_time(self, t_array):
        t_array = np.asarray(t_array)
        x = self.x + self.vx * t_array
        y = self.y + self.vy * t_array
        z = self.z + self.vz * t_array
        return np.column_stack((x, y, z))


def check_collision(xs, ys, zs, target, t_array=None, dt=0.01):
    """
    xs, ys, zs: missile trajectory arrays
    target: Target instance
    Returns: hit (bool), hit_index, closest_distance, closest_index
    """
    if t_array is None:
        t_array = np.arange(len(xs)) * dt
    else:
        t_array = np.asarray(t_array)

    target_positions = target.positions_over_time(t_array)
    distances = np.sqrt((xs - target_positions[:, 0])**2 +
                        (ys - target_positions[:, 1])**2 +
                        (zs - target_positions[:, 2])**2)

    hit_mask = distances <= target.radius
    closest_idx = int(np.argmin(distances))
    closest_distance = np.min(distances)

    if np.any(hit_mask):
        hit_index = int(np.argmax(hit_mask))
        return True, hit_index, closest_distance, closest_idx
    return False, None, closest_distance, closest_idx


def closest_approach_between_trajectories(traj1, traj2, t1, t2):
    """
    Compute closest Euclidean distance between two trajectories over shared time.

    traj1/traj2 shape: (N, 3)
    t1/t2 shape: (N,)
    """
    t_end = min(t1[-1], t2[-1])
    shared_t = np.linspace(0.0, t_end, 2000)

    p1 = np.column_stack([
        np.interp(shared_t, t1, traj1[:, 0]),
        np.interp(shared_t, t1, traj1[:, 1]),
        np.interp(shared_t, t1, traj1[:, 2]),
    ])
    p2 = np.column_stack([
        np.interp(shared_t, t2, traj2[:, 0]),
        np.interp(shared_t, t2, traj2[:, 1]),
        np.interp(shared_t, t2, traj2[:, 2]),
    ])

    distances = np.linalg.norm(p1 - p2, axis=1)
    idx = int(np.argmin(distances))
    return float(distances[idx]), float(shared_t[idx]), idx
