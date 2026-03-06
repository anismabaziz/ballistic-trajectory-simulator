import math
import time

import numpy as np


def _linspace(start, stop, count):
    if count <= 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [float(start + i * step) for i in range(count)]


def _acceleration(params, altitude_y, vx, vy, vz):
    wind_x = float(np.interp(altitude_y, params["alt_levels"], params["wind_x_vals"]))
    wind_z = float(np.interp(altitude_y, params["alt_levels"], params["wind_z_vals"]))
    wind_vertical = float(np.interp(altitude_y, params["alt_levels"], params["wind_vertical_vals"]))

    vx_rel = vx - wind_x
    vy_rel = vy - wind_vertical
    vz_rel = vz - wind_z
    v_rel = math.sqrt(vx_rel * vx_rel + vy_rel * vy_rel + vz_rel * vz_rel)

    if v_rel > 1e-9:
        drag_force = 0.5 * params["rho"] * params["drag_coefficient"] * params["area"] * (v_rel * v_rel)
        ax = -(drag_force / params["mass"]) * (vx_rel / v_rel)
        ay = -params["gravity"] - (drag_force / params["mass"]) * (vy_rel / v_rel)
        az = -(drag_force / params["mass"]) * (vz_rel / v_rel)
    else:
        ax, ay, az = 0.0, -params["gravity"], 0.0

    omega = 7.2921e-5
    lat = params["latitude"]
    ax += 2.0 * omega * vz * math.sin(lat)
    az += -2.0 * omega * vx * math.sin(lat)
    return ax, ay, az


def _simulate_candidate(params, speed, elevation_deg, azimuth_deg, target_x_start, target_vx, target_radius, dt=0.014):
    elev = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    vx = speed * math.cos(elev) * math.cos(az)
    vy = speed * math.sin(elev)
    vz = speed * math.cos(elev) * math.sin(az)
    x, y, z = 0.0, 0.0, 0.0
    t = 0.0

    max_time = max(6.0, min(45.0, (2.0 * speed * max(math.sin(elev), 0.04) / params["gravity"]) * 1.4 + 5.0))
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

        ax, ay, az_acc = _acceleration(params, y, vx, vy, vz)
        vx += ax * dt
        vy += ay * dt
        vz += az_acc * dt
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

    params = {
        "gravity": float(snapshot["gravity"]),
        "mass": float(snapshot["mass"]),
        "rho": float(snapshot["rho"]),
        "drag_coefficient": float(snapshot["drag_coefficient"]),
        "area": float(snapshot["area"]),
        "latitude": float(snapshot["latitude"]),
        "alt_levels": np.array(snapshot["alt_levels"], dtype=float),
        "wind_x_vals": np.array(snapshot["wind_x_vals"], dtype=float),
        "wind_z_vals": np.array(snapshot["wind_z_vals"], dtype=float),
        "wind_vertical_vals": np.array(snapshot["wind_vertical_vals"], dtype=float),
    }

    target_x_start = float(snapshot["target_x_launch"])
    target_vx = float(snapshot["target_velocity_x"])
    target_radius = float(snapshot["target_radius"])
    min_elev = max(2.0, float(snapshot["min_auto_elevation"]))

    best = None

    def better(cand, best_cand):
        if best_cand is None:
            return True
        if cand["hit"] != best_cand["hit"]:
            return cand["hit"]
        if cand["hit"]:
            return cand["hit_time"] < best_cand["hit_time"]
        return cand["distance"] < best_cand["distance"]

    t_max = min(50.0, max(18.0, abs(target_x_start) / 190.0 + 14.0))
    for t in _linspace(1.2, t_max, 54):
        if time.perf_counter() >= deadline:
            break
        x_t = target_x_start + target_vx * t
        if x_t <= 20.0:
            continue

        theta = math.degrees(math.atan((params["gravity"] * t * t) / (2.0 * x_t)))
        if theta < min_elev or theta > 87.5:
            continue

        denom = max(math.cos(math.radians(theta)) * t, 1e-6)
        speed = x_t / denom
        if speed < 70.0 or speed > 2200.0:
            continue

        for az in (-16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0):
            d, hit, hit_time = _simulate_candidate(params, speed, theta, az, target_x_start, target_vx, target_radius, dt=0.015)
            cand = {
                "distance": float(d),
                "hit": bool(hit),
                "speed": float(speed),
                "elevation": float(theta),
                "azimuth": float(az),
                "hit_time": float(hit_time) if hit_time is not None else 1e9,
            }
            if better(cand, best):
                best = cand

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
                d, hit, hit_time = _simulate_candidate(params, s, e, a, target_x_start, target_vx, target_radius, dt=0.012)
                cand = {
                    "distance": float(d),
                    "hit": bool(hit),
                    "speed": float(s),
                    "elevation": float(e),
                    "azimuth": float(a),
                    "hit_time": float(hit_time) if hit_time is not None else 1e9,
                }
                if better(cand, best):
                    best = cand

    return {
        "ok": True,
        "distance": best["distance"],
        "speed": best["speed"],
        "elevation": best["elevation"],
        "azimuth": best["azimuth"],
        "hit": best["hit"],
        "timed_out": time.perf_counter() >= deadline,
    }
