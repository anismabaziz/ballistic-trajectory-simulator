import math
import time

import numpy as np

from physics import BallisticPhysics


def _linspace(start, stop, count):
    if count <= 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [float(start + i * step) for i in range(count)]


def _simulate_candidate(physics, speed, elevation_deg, azimuth_deg, target_x_start, target_vx, target_radius, dt=0.014):
    elev = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    vx = speed * math.cos(elev) * math.cos(az)
    vy = speed * math.sin(elev)
    vz = speed * math.cos(elev) * math.sin(az)
    x, y, z = 0.0, 0.0, 0.0
    t = 0.0

    max_time = max(6.0, min(45.0, (2.0 * speed * max(math.sin(elev), 0.04) / physics.gravity) * 1.4 + 5.0))
    steps = int(max_time / dt)

    min_d = float("inf")
    hit = False
    hit_time = None
    for _ in range(steps):
        tx = target_x_start + target_vx * t
        d = math.sqrt((x - tx) ** 2 + y**2 + z**2)
        if d < min_d:
            min_d = d
        if d <= target_radius:
            hit = True
            hit_time = t
            break

        ax, ay, az_acceleration = physics.compute_acceleration(y, vx, vy, vz)
        vx += ax * dt
        vy += ay * dt
        vz += az_acceleration * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
        t += dt
        if y <= 0.0 and t > 0.2:
            break

    return min_d, hit, hit_time


def solve_launch(snapshot):
    start = time.perf_counter()
    deadline = start + float(snapshot.get("max_wall_s", 2.0))

    # The wind table is copied rather than referenced. `BallisticPhysics` keeps
    # whatever arrays it is handed with `asarray`, which does not copy a float
    # array, so passing the renderer's own arrays would let the atmosphere change
    # under the search while it solves. The problem being solved is fixed at launch.
    physics = BallisticPhysics(
        mass=snapshot["mass"],
        gravity=snapshot["gravity"],
        rho=snapshot["rho"],
        drag_coefficient=snapshot["drag_coefficient"],
        area=snapshot["area"],
        latitude=snapshot["latitude"],
        alt_levels=np.array(snapshot["alt_levels"], dtype=float),
        wind_x_vals=np.array(snapshot["wind_x_vals"], dtype=float),
        wind_z_vals=np.array(snapshot["wind_z_vals"], dtype=float),
        wind_vertical_vals=np.array(snapshot["wind_vertical_vals"], dtype=float),
    )

    target_x_start = float(snapshot["target_x_launch"])
    target_vx = float(snapshot["target_velocity_x"])
    target_radius = float(snapshot["target_radius"])
    min_elev = max(2.0, float(snapshot["min_auto_elevation"]))

    best = None

    def consider(speed, elevation, azimuth, d, hit, hit_time):
        """Keep the candidate if it beats the incumbent by the search's own ranking."""
        nonlocal best
        cand = {
            "distance": float(d),
            "hit": bool(hit),
            "speed": float(speed),
            "elevation": float(elevation),
            "azimuth": float(azimuth),
            "hit_time": float(hit_time) if hit_time is not None else 1e9,
        }
        if best is None:
            best = cand
        elif cand["hit"] != best["hit"]:
            if cand["hit"]:
                best = cand
        elif cand["hit"]:
            if cand["hit_time"] < best["hit_time"]:
                best = cand
        elif cand["distance"] < best["distance"]:
            best = cand

    t_max = min(50.0, max(18.0, abs(target_x_start) / 190.0 + 14.0))
    for t in _linspace(1.2, t_max, 54):
        if time.perf_counter() >= deadline:
            break
        x_t = target_x_start + target_vx * t
        if x_t <= 20.0:
            continue

        theta = math.degrees(math.atan((physics.gravity * t * t) / (2.0 * x_t)))
        if theta < min_elev or theta > 87.5:
            continue

        denom = max(math.cos(math.radians(theta)) * t, 1e-6)
        speed = x_t / denom
        if speed < 70.0 or speed > 2200.0:
            continue

        for az in (-16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0):
            consider(speed, theta, az, *_simulate_candidate(physics, speed, theta, az, target_x_start, target_vx, target_radius, dt=0.015))

    if best is None:
        return {"ok": False}

    refine_speeds = _linspace(max(70.0, best["speed"] - 100.0), min(2200.0, best["speed"] + 100.0), 7)
    refine_elev = _linspace(max(min_elev, best["elevation"] - 6.0), min(88.0, best["elevation"] + 6.0), 7)
    refine_az = _linspace(max(-35.0, best["azimuth"] - 6.0), min(35.0, best["azimuth"] + 6.0), 7)

    for s in refine_speeds:
        if time.perf_counter() >= deadline:
            break
        for e in refine_elev:
            for a in refine_az:
                consider(s, e, a, *_simulate_candidate(physics, s, e, a, target_x_start, target_vx, target_radius, dt=0.012))

    return {
        "ok": True,
        "distance": best["distance"],
        "speed": best["speed"],
        "elevation": best["elevation"],
        "azimuth": best["azimuth"],
        "hit": best["hit"],
        "timed_out": time.perf_counter() >= deadline,
    }
