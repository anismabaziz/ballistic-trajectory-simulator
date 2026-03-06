import math
import threading
import time

import numpy as np


class _Missile:
    def __init__(self, x, y, z, vx, vy, vz):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
        self.alive = True
        self.hit = False
        self.trail = [(self.x, self.y, self.z)]
        self.closest_distance = float("inf")


class _Slider:
    def __init__(self, name, minimum, maximum, value, rect):
        self.name = name
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.value = float(value)
        self.rect = rect
        self.dragging = False

    def normalized(self):
        span = max(self.maximum - self.minimum, 1e-9)
        return (self.value - self.minimum) / span

    def set_from_mouse(self, mouse_x):
        t = (mouse_x - self.rect.x) / max(self.rect.width, 1)
        t = max(0.0, min(1.0, t))
        self.value = self.minimum + t * (self.maximum - self.minimum)


class PygameBallisticSimulation:
    def __init__(
        self,
        simulator,
        launch_speed=300.0,
        launch_elevation_deg=35.0,
        launch_azimuth_deg=0.0,
        target_x=2800.0,
        target_radius=20.0,
        target_velocity_x=40.0,
        window_width=1280,
        window_height=720,
        time_scale=1.0,
    ):
        self.simulator = simulator
        self.launch_speed = float(launch_speed)
        self.launch_elevation_deg = float(launch_elevation_deg)
        self.launch_azimuth_deg = float(launch_azimuth_deg)
        self.target_x = float(target_x)
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_radius = float(target_radius)
        self.target_velocity_x = float(target_velocity_x)
        self.default_target_velocity_x = float(target_velocity_x)
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self.time_scale = float(time_scale)

        self.running = True
        self.sim_time = 0.0
        self.missiles = []
        self.world_lead_time = 2.0

        self.camera_distance = 4200.0
        self.camera_yaw_deg = 58.0
        self.camera_pitch_deg = 24.0
        self.camera_target_x = 1900.0
        self.camera_target_y = 260.0
        self.camera_target_z = 0.0
        self.camera_smooth = 0.12
        self._camera_yaw_velocity = 0.0
        self._camera_pitch_velocity = 0.0
        self._camera_pan_x_velocity = 0.0
        self._camera_pan_y_velocity = 0.0
        self._camera_pan_z_velocity = 0.0
        self._camera_zoom_velocity = 0.0

        self.base_rho = float(self.simulator.rho)
        self.base_drag = float(self.simulator.drag_coefficient)
        self.base_wind_x_vals = self.simulator.wind_x_vals.copy()
        self.base_wind_z_vals = self.simulator.wind_z_vals.copy()
        self.base_wind_vertical_vals = self.simulator.wind_vertical_vals.copy()

        self.wind_cross_scale = 1.0
        self.wind_vertical_bias = 0.0

        self.slider_panel_x = self.window_width - 330
        self.slider_panel_width = 330
        self.slider_panel_height = 520
        self.sliders = []
        self.auto_button_rect = None
        self.follow_button_rect = None
        self.target_mode_button_rect = None
        self.auto_status_text = ""
        self.auto_status_color = (160, 190, 175)
        self.auto_solver_thread = None
        self.auto_solver_running = False
        self.auto_solver_result = None
        self.auto_solver_started_sim_time = None

        self.target_hit_timer = 0.0

        self.camera_follow_target = True
        self.camera_follow_offset_x = -220.0
        self.camera_follow_offset_y = 160.0
        self.camera_follow_offset_z = 0.0

        self.current_mouse = (0, 0)
        self.orbit_dragging = False
        self.pan_dragging = False
        self.last_mouse_pos = None

    def reset(self):
        self.missiles = []
        self.sim_time = 0.0
        self.target_hit_timer = 0.0

    def _initialize_sliders(self, pygame):
        top = 80
        spacing = 44
        width = 220
        height = 16

        def rect_at(i):
            return pygame.Rect(self.slider_panel_x + 90, top + i * spacing, width, height)

        self.sliders = [
            _Slider("Wind X", 0.0, 40.0, float(self.base_wind_x_vals[-1]), rect_at(0)),
            _Slider("Wind Z", -30.0, 30.0, 0.0, rect_at(1)),
            _Slider("Vert Wind", -15.0, 15.0, 0.0, rect_at(2)),
            _Slider("Air Density", 0.7, 1.5, self.base_rho, rect_at(3)),
            _Slider("Drag Coeff", 0.1, 1.2, self.base_drag, rect_at(4)),
            _Slider("Time Scale", 0.2, 4.0, self.time_scale, rect_at(5)),
        ]
        self.auto_button_rect = pygame.Rect(self.slider_panel_x + 14, self.sliders[-1].rect.bottom + 20, 130, 30)
        self.follow_button_rect = pygame.Rect(self.slider_panel_x + 154, self.sliders[-1].rect.bottom + 20, 160, 30)
        self.target_mode_button_rect = pygame.Rect(self.slider_panel_x + 14, self.sliders[-1].rect.bottom + 58, 300, 30)

    def _toggle_target_mode(self):
        if abs(self.target_velocity_x) > 1e-9:
            self.default_target_velocity_x = self.target_velocity_x
            self.target_velocity_x = 0.0
        else:
            restored = self.default_target_velocity_x if abs(self.default_target_velocity_x) > 1e-9 else 40.0
            self.target_velocity_x = restored

    def _toggle_camera_follow(self):
        self.camera_follow_target = not self.camera_follow_target

    def _reset_camera_view(self):
        self.camera_distance = 4200.0
        self.camera_yaw_deg = 58.0
        self.camera_pitch_deg = 24.0
        self.camera_target_x = 1900.0
        self.camera_target_y = 260.0
        self.camera_target_z = 0.0
        self.camera_follow_offset_x = -220.0
        self.camera_follow_offset_y = 160.0
        self.camera_follow_offset_z = 0.0
        self._begin_camera_goals()

    def _linspace(self, start, stop, count):
        if count <= 1:
            return [float(start)]
        step = (stop - start) / (count - 1)
        return [float(start + step * i) for i in range(count)]

    def _simulate_candidate(self, speed, elevation_deg, azimuth_deg, dt):
        target_x_start = self._target_position()[0]
        elev = math.radians(elevation_deg)
        azim = math.radians(azimuth_deg)
        vx = speed * math.cos(elev) * math.cos(azim)
        vy = speed * math.sin(elev)
        vz = speed * math.cos(elev) * math.sin(azim)
        x, y, z = 0.0, 0.0, 0.0

        sim_t = 0.0
        max_time = max(10.0, min(120.0, (2.0 * speed * max(math.sin(elev), 0.05) / self.simulator.gravity) * 1.8 + 8.0))
        steps = int(max_time / dt)

        min_d = float("inf")
        hit = False
        for _ in range(steps):
            tx = target_x_start + self.target_velocity_x * sim_t
            dx = x - tx
            dy = y - self.target_y
            dz = z - self.target_z
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < min_d:
                min_d = d
            if d <= self.target_radius:
                hit = True
                break

            ax, ay, az = self.simulator.compute_acceleration(y, vx, vy, vz)
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            x += vx * dt
            y += vy * dt
            z += vz * dt
            sim_t += dt

            if y <= 0.0 and sim_t > 0.2:
                break

        return min_d, hit

    def _start_auto_solver(self):
        if self.auto_solver_running:
            return

        snapshot = {
            "target_x_launch": float(self._target_position()[0]),
            "target_y": float(self.target_y),
            "target_z": float(self.target_z),
            "target_radius": float(self.target_radius),
            "target_velocity_x": float(self.target_velocity_x),
            "gravity": float(self.simulator.gravity),
            "mass": float(self.simulator.mass),
            "rho": float(self.simulator.rho),
            "drag_coefficient": float(self.simulator.drag_coefficient),
            "area": float(self.simulator.area),
            "latitude": float(self.simulator.latitude),
            "alt_levels": np.array(self.simulator.alt_levels, dtype=float),
            "wind_x_vals": np.array(self.simulator.wind_x_vals, dtype=float),
            "wind_z_vals": np.array(self.simulator.wind_z_vals, dtype=float),
            "wind_vertical_vals": np.array(self.simulator.wind_vertical_vals, dtype=float),
            "time_scale": float(self.time_scale),
            "wall_start": float(time.perf_counter()),
        }

        self.auto_solver_running = True
        self.auto_solver_result = None
        self.auto_solver_started_sim_time = float(self.sim_time)
        self.auto_status_text = "Auto solving..."
        self.auto_status_color = (140, 200, 255)
        self.auto_solver_thread = threading.Thread(target=self._auto_solver_worker, args=(snapshot,), daemon=True)
        self.auto_solver_thread.start()

    def _simulate_candidate_for_target(self, speed, elevation_deg, azimuth_deg, target_x_start, dt=0.012):
        elev = math.radians(elevation_deg)
        azim = math.radians(azimuth_deg)
        vx = speed * math.cos(elev) * math.cos(azim)
        vy = speed * math.sin(elev)
        vz = speed * math.cos(elev) * math.sin(azim)
        x, y, z = 0.0, 0.0, 0.0
        t = 0.0

        max_time = max(8.0, min(90.0, (2.0 * speed * max(math.sin(elev), 0.05) / self.simulator.gravity) * 1.5 + 6.0))
        steps = int(max_time / dt)

        min_d = float("inf")
        hit = False
        hit_time = None
        for _ in range(steps):
            tx = target_x_start + self.target_velocity_x * t
            dx = x - tx
            dy = y - self.target_y
            dz = z - self.target_z
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            min_d = min(min_d, d)
            if d <= self.target_radius:
                hit = True
                hit_time = t
                break

            ax, ay, az = self.simulator.compute_acceleration(y, vx, vy, vz)
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            x += vx * dt
            y += vy * dt
            z += vz * dt
            t += dt
            if y <= 0.0 and t > 0.2:
                break

        return min_d, hit, hit_time

    def _auto_solver_worker(self, snapshot):
        omega = 7.2921e-5

        def accel(y, vx, vy, vz):
            wind_x = np.interp(y, snapshot["alt_levels"], snapshot["wind_x_vals"])
            wind_z = np.interp(y, snapshot["alt_levels"], snapshot["wind_z_vals"])
            wind_vertical = np.interp(y, snapshot["alt_levels"], snapshot["wind_vertical_vals"])

            vx_rel = vx - wind_x
            vy_rel = vy - wind_vertical
            vz_rel = vz - wind_z
            v_rel = math.sqrt(vx_rel * vx_rel + vy_rel * vy_rel + vz_rel * vz_rel)

            if v_rel > 1e-9:
                drag_force = 0.5 * snapshot["rho"] * snapshot["drag_coefficient"] * snapshot["area"] * (v_rel * v_rel)
                ax = -(drag_force / snapshot["mass"]) * (vx_rel / v_rel)
                ay = -snapshot["gravity"] - (drag_force / snapshot["mass"]) * (vy_rel / v_rel)
                az = -(drag_force / snapshot["mass"]) * (vz_rel / v_rel)
            else:
                ax, ay, az = 0.0, -snapshot["gravity"], 0.0

            ax += 2.0 * omega * vz * math.sin(snapshot["latitude"])
            az += -2.0 * omega * vx * math.sin(snapshot["latitude"])
            return ax, ay, az

        def simulate_candidate(speed, elevation_deg, azimuth_deg, dt, target_x_start=None):
            elev = math.radians(elevation_deg)
            azim = math.radians(azimuth_deg)
            vx = speed * math.cos(elev) * math.cos(azim)
            vy = speed * math.sin(elev)
            vz = speed * math.cos(elev) * math.sin(azim)
            x, y, z = 0.0, 0.0, 0.0
            sim_t = 0.0

            if target_x_start is None:
                target_x_start = snapshot["target_x_launch"]

            max_time = max(8.0, min(90.0, (2.0 * speed * max(math.sin(elev), 0.05) / snapshot["gravity"]) * 1.5 + 6.0))
            steps = int(max_time / dt)

            min_d = float("inf")
            hit = False
            hit_time = None
            for _ in range(steps):
                tx = target_x_start + snapshot["target_velocity_x"] * sim_t
                dx = x - tx
                dy = y - snapshot["target_y"]
                dz = z - snapshot["target_z"]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d < min_d:
                    min_d = d
                if d <= snapshot["target_radius"]:
                    hit = True
                    hit_time = sim_t
                    break

                ax, ay, az = accel(y, vx, vy, vz)
                vx += ax * dt
                vy += ay * dt
                vz += az * dt
                x += vx * dt
                y += vy * dt
                z += vz * dt
                sim_t += dt
                if y <= 0.0 and sim_t > 0.2:
                    break

            return min_d, hit, hit_time

        def better(cand, best):
            if best is None:
                return True
            # Prefer actual hits first.
            if cand[0] != best[0]:
                return cand[0] < best[0]
            # Among hits, prefer earlier intercept, then smaller miss distance.
            if cand[0] == 0:
                if cand[1] != best[1]:
                    return cand[1] < best[1]
                if cand[2] != best[2]:
                    return cand[2] < best[2]
                return cand[3] < best[3]
            # Among misses, minimize miss distance then speed.
            if cand[2] != best[2]:
                return cand[2] < best[2]
            return cand[3] < best[3]

        coarse_speeds = self._linspace(120.0, 1450.0, 17)
        coarse_elev = self._linspace(4.0, 84.0, 17)
        coarse_az = self._linspace(-20.0, 20.0, 13)

        # (priority, hit_time, miss_dist, speed, elev, az)
        best = None
        counter = 0
        for s in coarse_speeds:
            for e in coarse_elev:
                for a in coarse_az:
                    d, hit, hit_t = simulate_candidate(s, e, a, dt=0.03)
                    priority = 0 if hit else 1
                    hit_t_cmp = hit_t if hit_t is not None else 1e9
                    cand = (priority, hit_t_cmp, d, s, e, a)
                    if better(cand, best):
                        best = cand
                    counter += 1
                    if counter % 100 == 0:
                        time.sleep(0)

        if best is None:
            self.auto_solver_result = {"ok": False}
            return

        _, _, _, s0, e0, a0 = best
        speed_steps = [90.0, 45.0, 22.0, 10.0, 5.0]
        angle_steps = [6.0, 3.0, 1.6, 0.8, 0.4]
        az_steps = [5.0, 2.5, 1.2, 0.6, 0.3]

        cur = best
        counter = 0
        for i in range(len(speed_steps)):
            s_step = speed_steps[i]
            e_step = angle_steps[i]
            a_step = az_steps[i]
            base_s, base_e, base_a = cur[3], cur[4], cur[5]
            for ds in (-s_step, 0.0, s_step):
                for de in (-e_step, 0.0, e_step):
                    for da in (-a_step, 0.0, a_step):
                        s = min(1500.0, max(80.0, base_s + ds))
                        e = min(88.0, max(2.0, base_e + de))
                        a = min(30.0, max(-30.0, base_a + da))
                        d, hit, hit_t = simulate_candidate(s, e, a, dt=0.012)
                        priority = 0 if hit else 1
                        hit_t_cmp = hit_t if hit_t is not None else 1e9
                        cand = (priority, hit_t_cmp, d, s, e, a)
                        if better(cand, cur):
                            cur = cand
                        counter += 1
                        if counter % 120 == 0:
                            time.sleep(0)

        wall_elapsed = max(0.0, float(time.perf_counter()) - snapshot["wall_start"])
        sim_advance = wall_elapsed * max(snapshot["time_scale"], 0.05)
        predicted_target_x = snapshot["target_x_launch"] + snapshot["target_velocity_x"] * sim_advance

        refine = cur
        for ds in (-35.0, -15.0, 0.0, 15.0, 35.0):
            for de in (-2.0, -1.0, 0.0, 1.0, 2.0):
                for da in (-1.5, -0.7, 0.0, 0.7, 1.5):
                    s = min(1500.0, max(80.0, cur[3] + ds))
                    e = min(88.0, max(2.0, cur[4] + de))
                    a = min(30.0, max(-30.0, cur[5] + da))
                    d, hit, hit_t = simulate_candidate(
                        s,
                        e,
                        a,
                        dt=0.009,
                        target_x_start=predicted_target_x,
                    )

                    priority = 0 if hit else 1
                    hit_t_cmp = hit_t if hit_t is not None else 1e9
                    cand = (priority, hit_t_cmp, d, s, e, a)
                    if better(cand, refine):
                        refine = cand

        cur = refine

        self.auto_solver_result = {
            "ok": True,
            "distance": cur[2],
            "speed": cur[3],
            "elevation": cur[4],
            "azimuth": cur[5],
            "hit": cur[0] == 0,
        }

    def _update_auto_solver(self):
        if not self.auto_solver_running:
            return
        if self.auto_solver_result is None:
            return

        result = self.auto_solver_result
        self.auto_solver_result = None
        self.auto_solver_running = False

        if not result.get("ok"):
            self.auto_status_text = "Auto solve failed"
            self.auto_status_color = (255, 140, 140)
            self.auto_solver_started_sim_time = None
            return

        best_speed = result["speed"]
        best_angle = result["elevation"]
        best_azimuth = result["azimuth"]

        elapsed_sim = 0.0
        if self.auto_solver_started_sim_time is not None:
            elapsed_sim = max(0.0, self.sim_time - self.auto_solver_started_sim_time)
        predicted_target_x = self._target_position()[0] + self.target_velocity_x * min(0.35, elapsed_sim * 0.25)

        local_best = (result["distance"], best_speed, best_angle, best_azimuth, result["hit"], 1e9)
        for ds in (-80.0, -40.0, 0.0, 40.0, 80.0):
            for de in (-4.0, -2.0, 0.0, 2.0, 4.0):
                for da in (-3.0, -1.5, 0.0, 1.5, 3.0):
                    s = min(1500.0, max(80.0, best_speed + ds))
                    e = min(88.0, max(2.0, best_angle + de))
                    a = min(30.0, max(-30.0, best_azimuth + da))
                    d, h, ht = self._simulate_candidate_for_target(s, e, a, predicted_target_x, dt=0.009)
                    ht_cmp = ht if ht is not None else 1e9
                    if h and not local_best[4]:
                        local_best = (d, s, e, a, h, ht_cmp)
                    elif h and local_best[4] and ht_cmp < local_best[5]:
                        local_best = (d, s, e, a, h, ht_cmp)
                    elif (not local_best[4]) and d < local_best[0]:
                        local_best = (d, s, e, a, h, ht_cmp)

        best_d, best_speed, best_angle, best_azimuth, _, _ = local_best
        self.auto_solver_started_sim_time = None

        self.launch_speed = best_speed
        self.launch_elevation_deg = best_angle
        self.launch_azimuth_deg = best_azimuth
        self._launch_missile()

        if best_d <= self.target_radius:
            self.auto_status_text = (
                f"Auto launched hit: {best_speed:.1f}m/s, elev {best_angle:.1f}, az {best_azimuth:.1f}"
            )
            self.auto_status_color = (120, 240, 150)
        else:
            self.auto_status_text = (
                f"Auto launched best: {best_speed:.1f}m/s, elev {best_angle:.1f}, az {best_azimuth:.1f}, miss {best_d:.1f}m"
            )
            self.auto_status_color = (255, 210, 130)

    def _apply_slider_values(self):
        wind_x_top = self.sliders[0].value
        wind_z_top = self.sliders[1].value
        vertical_bias = self.sliders[2].value
        self.simulator.rho = self.sliders[3].value
        self.simulator.drag_coefficient = self.sliders[4].value
        self.time_scale = self.sliders[5].value

        max_alt = max(float(self.simulator.alt_levels[-1]), 1.0)
        alt_norm = self.simulator.alt_levels / max_alt

        self.simulator.wind_x_vals = self.base_wind_x_vals + alt_norm * wind_x_top
        self.simulator.wind_z_vals = self.base_wind_z_vals + alt_norm * wind_z_top
        self.simulator.wind_vertical_vals = self.base_wind_vertical_vals + vertical_bias

    def _launch_missile(self):
        elev = math.radians(self.launch_elevation_deg)
        azim = math.radians(self.launch_azimuth_deg)
        vx = self.launch_speed * math.cos(elev) * math.cos(azim)
        vy = self.launch_speed * math.sin(elev)
        vz = self.launch_speed * math.cos(elev) * math.sin(azim)
        self.missiles.append(_Missile(0.0, 0.0, 0.0, vx, vy, vz))

    def _target_position(self, t=None):
        tt = self.sim_time if t is None else float(t)
        x = self.target_x + self.target_velocity_x * tt
        return x, self.target_y, self.target_z

    def _update_physics(self, dt):
        self.sim_time += dt
        self.target_hit_timer = max(0.0, self.target_hit_timer - dt)
        tx, ty, tz = self._target_position()

        for missile in self.missiles:
            if missile.alive:
                ax, ay, az = self.simulator.compute_acceleration(missile.y, missile.vx, missile.vy, missile.vz)
                missile.vx += ax * dt
                missile.vy += ay * dt
                missile.vz += az * dt

                missile.x += missile.vx * dt
                missile.y += missile.vy * dt
                missile.z += missile.vz * dt

                if missile.y <= 0.0 and len(missile.trail) > 3:
                    missile.y = 0.0
                    missile.alive = False

            dx = missile.x - tx
            dy = missile.y - ty
            dz = missile.z - tz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            missile.closest_distance = min(missile.closest_distance, d)
            if d <= self.target_radius and missile.alive:
                missile.hit = True
                missile.alive = False
                self.target_hit_timer = 1.2

            missile.trail.append((missile.x, missile.y, missile.z))
            if len(missile.trail) > 1200:
                missile.trail = missile.trail[-1200:]

    def _camera_basis(self):
        yaw = math.radians(self.camera_yaw_deg)
        pitch = math.radians(self.camera_pitch_deg)

        cx = self.camera_target_x + self.camera_distance * math.cos(pitch) * math.cos(yaw)
        cy = self.camera_target_y + self.camera_distance * math.sin(pitch)
        cz = self.camera_target_z + self.camera_distance * math.cos(pitch) * math.sin(yaw)

        fx = self.camera_target_x - cx
        fy = self.camera_target_y - cy
        fz = self.camera_target_z - cz
        fl = max(math.sqrt(fx * fx + fy * fy + fz * fz), 1e-9)
        fx, fy, fz = fx / fl, fy / fl, fz / fl

        upx, upy, upz = 0.0, 1.0, 0.0
        rx = fz * upy - fy * upz
        ry = fx * upz - fz * upx
        rz = fy * upx - fx * upy
        rl = max(math.sqrt(rx * rx + ry * ry + rz * rz), 1e-9)
        rx, ry, rz = rx / rl, ry / rl, rz / rl

        ux = fy * rz - fz * ry
        uy = fz * rx - fx * rz
        uz = fx * ry - fy * rx

        return (cx, cy, cz), (rx, ry, rz), (ux, uy, uz), (fx, fy, fz)

    def _project(self, x, y, z):
        (cx, cy, cz), right, up, forward = self._camera_basis()
        dx = x - cx
        dy = y - cy
        dz = z - cz

        cam_x = dx * right[0] + dy * right[1] + dz * right[2]
        cam_y = dx * up[0] + dy * up[1] + dz * up[2]
        cam_z = dx * forward[0] + dy * forward[1] + dz * forward[2]

        if cam_z <= 1.0:
            return None

        focal = 920.0
        sx = self.window_width * 0.5 + (cam_x / cam_z) * focal
        sy = self.window_height * 0.5 - (cam_y / cam_z) * focal
        return int(sx), int(sy), cam_z

    def _draw_ground_grid(self, screen, pygame):
        grid_spacing = 500
        x_min = -1000
        x_max = 10000
        z_min = -3000
        z_max = 3000

        for gx in range(x_min, x_max + 1, grid_spacing):
            p1 = self._project(gx, 0.0, z_min)
            p2 = self._project(gx, 0.0, z_max)
            if p1 and p2:
                pygame.draw.line(screen, (72, 92, 76), (p1[0], p1[1]), (p2[0], p2[1]), 1)

        for gz in range(z_min, z_max + 1, grid_spacing):
            p1 = self._project(x_min, 0.0, gz)
            p2 = self._project(x_max, 0.0, gz)
            if p1 and p2:
                pygame.draw.line(screen, (72, 92, 76), (p1[0], p1[1]), (p2[0], p2[1]), 1)

        for marker_x in range(0, x_max + 1, 1000):
            p = self._project(marker_x, 0.0, 0.0)
            if p:
                label = self._font_small.render(f"{marker_x}m", True, (170, 190, 165))
                screen.blit(label, (p[0] + 4, p[1] - 10))

    def _draw_altitude_grid(self, screen, pygame):
        for h in range(250, 2001, 250):
            p1 = self._project(0.0, h, -2200)
            p2 = self._project(0.0, h, 2200)
            if p1 and p2:
                pygame.draw.line(screen, (90, 110, 130), (p1[0], p1[1]), (p2[0], p2[1]), 1)
                text = self._font_small.render(f"{h}m", True, (140, 180, 220))
                screen.blit(text, (p1[0] + 8, p1[1] - 14))

    def _draw_target(self, screen, pygame):
        tx, ty, tz = self._target_position()
        body = self._project(tx, ty + self.target_radius * 0.8, tz)
        base = self._project(tx, ty, tz)
        if not body or not base:
            return

        radius_px = max(4, int(2200 * self.target_radius / max(body[2], 1.0) / 30.0))
        is_hit = self.target_hit_timer > 0.0
        fill_col = (250, 86, 86) if is_hit else (70, 220, 100)
        edge_col = (148, 20, 20) if is_hit else (36, 110, 60)
        pygame.draw.circle(screen, fill_col, (body[0], body[1]), radius_px, 0)
        pygame.draw.circle(screen, edge_col, (body[0], body[1]), radius_px, 2)
        pygame.draw.line(screen, (50, 120, 60), (body[0], body[1]), (base[0], base[1]), 2)

        if is_hit:
            hit_text = self._font_small.render("HIT", True, (255, 215, 215))
            screen.blit(hit_text, (body[0] + 10, body[1] - 10))

        lead_t = self.world_lead_time
        lx, ly, lz = self._target_position(self.sim_time + lead_t)
        lead = self._project(lx, ly, lz)
        if lead:
            pygame.draw.circle(screen, (255, 230, 120), (lead[0], lead[1]), 7, 2)
            pygame.draw.line(screen, (255, 230, 120), (body[0], body[1]), (lead[0], lead[1]), 1)
            txt = self._font_small.render("lead", True, (255, 230, 120))
            screen.blit(txt, (lead[0] + 8, lead[1] - 8))

    def _draw_missile_shape(self, screen, pygame, missile):
        def v_add(a, b):
            return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

        def v_sub(a, b):
            return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

        def v_scale(v, s):
            return (v[0] * s, v[1] * s, v[2] * s)

        def v_dot(a, b):
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        def v_cross(a, b):
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        def v_norm(v):
            l = math.sqrt(max(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], 1e-12))
            return (v[0] / l, v[1] / l, v[2] / l)

        center = (missile.x, missile.y, missile.z)
        vel = (missile.vx, missile.vy, missile.vz)
        speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
        forward = v_norm(vel) if speed > 1e-6 else (1.0, 0.0, 0.0)

        ref_up = (0.0, 1.0, 0.0)
        if abs(v_dot(forward, ref_up)) > 0.9:
            ref_up = (0.0, 0.0, 1.0)

        right = v_norm(v_cross(forward, ref_up))
        up = v_norm(v_cross(right, forward))

        body_len = 32.0
        nose_len = 12.0
        body_radius = 2.2
        ring_count = 8

        tail_center = v_sub(center, v_scale(forward, body_len * 0.5))
        nose_base_center = v_add(center, v_scale(forward, body_len * 0.5))
        tip = v_add(nose_base_center, v_scale(forward, nose_len))

        tail_ring = []
        nose_ring = []
        for i in range(ring_count):
            a = (2.0 * math.pi * i) / ring_count
            radial = v_add(v_scale(right, math.cos(a) * body_radius), v_scale(up, math.sin(a) * body_radius))
            tail_ring.append(v_add(tail_center, radial))
            nose_ring.append(v_add(nose_base_center, radial))

        polygons = []

        def add_polygon(points3d, color):
            projected_points = []
            for p in points3d:
                proj = self._project(p[0], p[1], p[2])
                if proj is None:
                    return
                px, py, pz = proj
                projected_points.append((px, py, pz))

            depth = sum(item[2] for item in projected_points) / len(projected_points)
            points2d = [(item[0], item[1]) for item in projected_points]
            polygons.append((depth, points2d, color))

        body_color = (150, 160, 175) if not missile.hit else (120, 210, 145)
        nose_color = (210, 215, 225) if not missile.hit else (190, 255, 210)
        fin_color = (90, 110, 130) if not missile.hit else (75, 170, 105)

        for i in range(ring_count):
            j = (i + 1) % ring_count
            q = [tail_ring[i], tail_ring[j], nose_ring[j], nose_ring[i]]
            shade = 0.74 + 0.22 * (i / max(ring_count - 1, 1))
            col = (
                int(body_color[0] * shade),
                int(body_color[1] * shade),
                int(body_color[2] * shade),
            )
            add_polygon(q, col)

        for i in range(ring_count):
            j = (i + 1) % ring_count
            tri = [nose_ring[i], nose_ring[j], tip]
            shade = 0.82 + 0.18 * (i / max(ring_count - 1, 1))
            col = (
                int(nose_color[0] * shade),
                int(nose_color[1] * shade),
                int(nose_color[2] * shade),
            )
            add_polygon(tri, col)

        fin_base = v_sub(tail_center, v_scale(forward, 4.0))
        for angle_deg in (45.0, 135.0, 225.0, 315.0):
            a = math.radians(angle_deg)
            radial = v_add(v_scale(right, math.cos(a)), v_scale(up, math.sin(a)))
            p1 = v_add(fin_base, v_scale(radial, body_radius * 0.95))
            p2 = v_add(fin_base, v_scale(radial, body_radius + 4.6))
            p3 = v_sub(p2, v_scale(forward, 8.0))
            p4 = v_sub(p1, v_scale(forward, 7.0))
            add_polygon([p1, p2, p3, p4], fin_color)

        polygons.sort(key=lambda item: item[0], reverse=True)
        for _, points2d, color in polygons:
            pygame.draw.polygon(screen, color, points2d)

        tip2d = self._project(tip[0], tip[1], tip[2])
        if tip2d:
            pygame.draw.circle(screen, (255, 245, 230), (tip2d[0], tip2d[1]), 2)

    def _draw_missiles(self, screen, pygame):
        for missile in self.missiles:
            if len(missile.trail) > 1:
                pts = []
                for p in missile.trail[-220:]:
                    pr = self._project(p[0], p[1], p[2])
                    if pr:
                        pts.append((pr[0], pr[1]))
                if len(pts) >= 2:
                    col = (120, 255, 165) if missile.hit else (255, 205, 120)
                    pygame.draw.lines(screen, col, False, pts, 2)

            self._draw_missile_shape(screen, pygame, missile)

    def _draw_wrapped_text(self, screen, font, text, color, x, y, max_width, line_height):
        words = text.split()
        line = ""
        out_y = y
        for w in words:
            trial = w if not line else f"{line} {w}"
            if font.size(trial)[0] <= max_width:
                line = trial
            else:
                screen.blit(font.render(line, True, color), (x, out_y))
                out_y += line_height
                line = w
        if line:
            screen.blit(font.render(line, True, color), (x, out_y))
            out_y += line_height
        return out_y

    def _draw_slider_panel(self, screen, pygame):
        panel_w = self.slider_panel_width
        panel_h = self.slider_panel_height
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((16, 22, 30, 206))
        screen.blit(panel, (self.slider_panel_x, 24))

        title = self._font_title.render("Environment Controls", True, (220, 230, 250))
        screen.blit(title, (self.slider_panel_x + 14, 35))

        for idx, slider in enumerate(self.sliders):
            y = 64 + idx * 44
            lbl = self._font_small.render(f"{slider.name}: {slider.value:.2f}", True, (176, 208, 255))
            screen.blit(lbl, (self.slider_panel_x + 14, y))

            pygame.draw.rect(screen, (68, 84, 104), slider.rect, border_radius=6)
            fill_rect = pygame.Rect(slider.rect.x, slider.rect.y, int(slider.rect.width * slider.normalized()), slider.rect.height)
            pygame.draw.rect(screen, (108, 178, 255), fill_rect, border_radius=6)
            handle_x = slider.rect.x + int(slider.rect.width * slider.normalized())
            pygame.draw.circle(screen, (235, 242, 255), (handle_x, slider.rect.y + slider.rect.height // 2), 7)

        if self.auto_button_rect is not None:
            button_color = (62, 132, 210) if not self.auto_solver_running else (95, 95, 115)
            pygame.draw.rect(screen, button_color, self.auto_button_rect, border_radius=8)
            pygame.draw.rect(screen, (110, 170, 235), self.auto_button_rect, width=2, border_radius=8)
            button_name = "Auto Solve" if not self.auto_solver_running else "Solving..."
            button_label = self._font_small.render(button_name, True, (240, 246, 255))
            screen.blit(button_label, (self.auto_button_rect.x + 25, self.auto_button_rect.y + 7))

        if self.follow_button_rect is not None:
            follow_color = (68, 150, 110) if self.camera_follow_target else (110, 88, 88)
            pygame.draw.rect(screen, follow_color, self.follow_button_rect, border_radius=8)
            pygame.draw.rect(screen, (190, 220, 205), self.follow_button_rect, width=2, border_radius=8)
            follow_text = "Cam Follow: ON" if self.camera_follow_target else "Cam Follow: OFF"
            screen.blit(self._font_small.render(follow_text, True, (245, 248, 255)), (self.follow_button_rect.x + 12, self.follow_button_rect.y + 7))

        if self.target_mode_button_rect is not None:
            moving = abs(self.target_velocity_x) > 1e-9
            mode_color = (72, 138, 205) if moving else (108, 108, 120)
            pygame.draw.rect(screen, mode_color, self.target_mode_button_rect, border_radius=8)
            pygame.draw.rect(screen, (176, 208, 246), self.target_mode_button_rect, width=2, border_radius=8)
            mode_text = f"Target Mode: {'Moving' if moving else 'Stationary'} (toggle)"
            screen.blit(self._font_small.render(mode_text, True, (242, 247, 255)), (self.target_mode_button_rect.x + 12, self.target_mode_button_rect.y + 7))

        if self.auto_status_text:
            status_x = self.slider_panel_x + 14
            anchor_y = 320
            target_mode_rect = self.target_mode_button_rect
            auto_rect = self.auto_button_rect
            if target_mode_rect is not None:
                anchor_y = target_mode_rect.bottom + 10
            elif auto_rect is not None:
                anchor_y = auto_rect.bottom + 8
            status_end_y = self._draw_wrapped_text(
                screen,
                self._font_small,
                self.auto_status_text,
                self.auto_status_color,
                status_x,
                anchor_y,
                panel_w - 28,
                16,
            )
        else:
            status_end_y = (self.target_mode_button_rect.bottom + 8) if self.target_mode_button_rect is not None else 320

        help_lines = [
            "Mouse L-drag orbit, R-drag pan, wheel zoom.",
            "IJKLUO camera keys, WASD pan keys.",
            "C toggle camera-follow, M target mode toggle.",
            "SPACE launch, R reset, V camera reset.",
            "Arrows speed/elevation, Q/E azimuth, T/G target vx.",
        ]
        target_mode_rect = self.target_mode_button_rect
        y = max((target_mode_rect.bottom + 48) if target_mode_rect is not None else 320, status_end_y + 8)
        max_text_width = panel_w - 28
        for line in help_lines:
            y = self._draw_wrapped_text(
                screen,
                self._font_small,
                line,
                (160, 190, 175),
                self.slider_panel_x + 14,
                y,
                max_text_width,
                16,
            )

    def _draw_hud(self, screen, pygame):
        panel = pygame.Surface((620, 188), pygame.SRCALPHA)
        panel.fill((9, 14, 22, 196))
        screen.blit(panel, (12, 12))

        active = sum(1 for m in self.missiles if m.alive)
        hits = sum(1 for m in self.missiles if m.hit)
        closest = min((m.closest_distance for m in self.missiles), default=float("inf"))
        closest_text = "N/A" if math.isinf(closest) else f"{closest:.2f} m"

        lines = [
            "Interactive 3D Ballistic Simulator",
            f"Launch: {self.launch_speed:.1f} m/s | Elev {self.launch_elevation_deg:.1f} deg | Azi {self.launch_azimuth_deg:.1f} deg",
            f"Target: x0 {self.target_x:.0f} m | vx {self.target_velocity_x:.1f} m/s ({'moving' if abs(self.target_velocity_x) > 1e-9 else 'stationary'}) | radius {self.target_radius:.1f} m",
            f"Target state: {'HIT' if self.target_hit_timer > 0.0 else 'ACTIVE'}",
            f"Missiles: total {len(self.missiles)} | active {active} | hits {hits}",
            f"Closest approach: {closest_text} | Sim time: {self.sim_time:.2f} s",
        ]

        y = 22
        for idx, line in enumerate(lines):
            col = (238, 243, 255) if idx == 0 else (194, 210, 234)
            text_surface = self._font_main.render(line, True, col)
            screen.blit(text_surface, (24, y))
            y += 33

        controls = "Controls: SPACE launch | R reset | Arrows speed/elevation | Q/E azimuth | T/G target speed"
        controls_2 = "C follow-cam | M target mode | V camera reset | SHIFT = faster camera"
        screen.blit(self._font_small.render(controls, True, (148, 196, 170)), (24, 174))
        screen.blit(self._font_small.render(controls_2, True, (140, 182, 160)), (24, 192))

    def _is_over_slider_panel(self, pos):
        mx, my = pos
        return (
            self.slider_panel_x <= mx <= self.slider_panel_x + self.slider_panel_width
            and 24 <= my <= 24 + self.slider_panel_height
        )

    def _apply_camera_follow(self, dt):
        if not self.camera_follow_target:
            return
        tx, ty, tz = self._target_position()
        lead_t = 0.8
        tx = tx + self.target_velocity_x * lead_t

        desired_x = tx + self.camera_follow_offset_x
        desired_y = max(40.0, ty + self.camera_follow_offset_y)
        desired_z = tz + self.camera_follow_offset_z

        k = min(1.0, dt * 4.5)
        self._camera_target_x_goal += (desired_x - self._camera_target_x_goal) * k
        self._camera_target_y_goal += (desired_y - self._camera_target_y_goal) * k
        self._camera_target_z_goal += (desired_z - self._camera_target_z_goal) * k

    def _handle_mouse_camera(self, event, pygame):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and not self._is_over_slider_panel(event.pos):
                self.orbit_dragging = True
                self.last_mouse_pos = event.pos
            elif event.button == 3 and not self._is_over_slider_panel(event.pos):
                self.pan_dragging = True
                self.last_mouse_pos = event.pos
            elif event.button == 4:
                self._camera_distance_goal = max(600.0, self._camera_distance_goal * 0.92)
            elif event.button == 5:
                self._camera_distance_goal = min(16000.0, self._camera_distance_goal * 1.08)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.orbit_dragging = False
            elif event.button == 3:
                self.pan_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.last_mouse_pos is not None:
            dx = event.pos[0] - self.last_mouse_pos[0]
            dy = event.pos[1] - self.last_mouse_pos[1]
            self.last_mouse_pos = event.pos

            if self.orbit_dragging:
                self._camera_yaw_goal += dx * 0.22
                self._camera_pitch_goal = max(-8.0, min(85.0, self._camera_pitch_goal - dy * 0.18))
            elif self.pan_dragging:
                pan_scale = 1.8
                if self.camera_follow_target:
                    self.camera_follow_offset_x -= dx * pan_scale
                    self.camera_follow_offset_y += dy * pan_scale
                else:
                    self._camera_target_x_goal -= dx * pan_scale
                    self._camera_target_y_goal += dy * pan_scale

    def _axis_input(self, negative_pressed, positive_pressed):
        value = 0.0
        if negative_pressed:
            value -= 1.0
        if positive_pressed:
            value += 1.0
        return value

    def _smooth_velocity(self, current, target, dt, rise=8.0, decay=9.0):
        if abs(target) > 1e-6:
            k = min(1.0, rise * dt)
            return current + (target - current) * k
        k = min(1.0, decay * dt)
        return current * (1.0 - k)

    def _draw_scene(self, screen, pygame):
        for row in range(self.window_height):
            t = row / max(self.window_height - 1, 1)
            r = int(11 * (1 - t) + 32 * t)
            g = int(19 * (1 - t) + 60 * t)
            b = int(34 * (1 - t) + 98 * t)
            pygame.draw.line(screen, (r, g, b), (0, row), (self.window_width, row))

        self._draw_ground_grid(screen, pygame)
        self._draw_altitude_grid(screen, pygame)
        self._draw_target(screen, pygame)
        self._draw_missiles(screen, pygame)
        self._draw_hud(screen, pygame)
        self._draw_slider_panel(screen, pygame)

    def _update_camera_smooth(self, dt):
        blend = 1.0 - (1.0 - self.camera_smooth) ** max(dt * 60.0, 0.0)
        self.camera_distance += (self._camera_distance_goal - self.camera_distance) * blend
        self.camera_yaw_deg += (self._camera_yaw_goal - self.camera_yaw_deg) * blend
        self.camera_pitch_deg += (self._camera_pitch_goal - self.camera_pitch_deg) * blend
        self.camera_target_x += (self._camera_target_x_goal - self.camera_target_x) * blend
        self.camera_target_y += (self._camera_target_y_goal - self.camera_target_y) * blend
        self.camera_target_z += (self._camera_target_z_goal - self.camera_target_z) * blend

    def _begin_camera_goals(self):
        self._camera_distance_goal = self.camera_distance
        self._camera_yaw_goal = self.camera_yaw_deg
        self._camera_pitch_goal = self.camera_pitch_deg
        self._camera_target_x_goal = self.camera_target_x
        self._camera_target_y_goal = self.camera_target_y
        self._camera_target_z_goal = self.camera_target_z

    def _handle_pressed_keys(self, keys, dt, pygame):
        speed_scale = 2.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0
        yaw_input = self._axis_input(keys[pygame.K_j], keys[pygame.K_l])
        pitch_input = self._axis_input(keys[pygame.K_k], keys[pygame.K_i])
        zoom_input = self._axis_input(keys[pygame.K_o], keys[pygame.K_u])
        pan_x_input = self._axis_input(keys[pygame.K_a], keys[pygame.K_d])
        pan_z_input = self._axis_input(keys[pygame.K_w], keys[pygame.K_s])

        self._camera_yaw_velocity = self._smooth_velocity(self._camera_yaw_velocity, yaw_input * 70.0 * speed_scale, dt)
        self._camera_pitch_velocity = self._smooth_velocity(self._camera_pitch_velocity, pitch_input * 55.0 * speed_scale, dt)
        self._camera_zoom_velocity = self._smooth_velocity(self._camera_zoom_velocity, zoom_input * 1200.0 * speed_scale, dt)
        self._camera_pan_x_velocity = self._smooth_velocity(self._camera_pan_x_velocity, pan_x_input * 520.0 * speed_scale, dt)
        self._camera_pan_z_velocity = self._smooth_velocity(self._camera_pan_z_velocity, pan_z_input * 520.0 * speed_scale, dt)

        self._camera_yaw_goal += self._camera_yaw_velocity * dt
        self._camera_pitch_goal = max(-10.0, min(80.0, self._camera_pitch_goal + self._camera_pitch_velocity * dt))
        self._camera_distance_goal = max(600.0, min(16000.0, self._camera_distance_goal + self._camera_zoom_velocity * dt))
        if self.camera_follow_target:
            self.camera_follow_offset_x += self._camera_pan_x_velocity * dt
            self.camera_follow_offset_z += self._camera_pan_z_velocity * dt
        else:
            self._camera_target_x_goal += self._camera_pan_x_velocity * dt
            self._camera_target_z_goal += self._camera_pan_z_velocity * dt

    def _handle_slider_events(self, event, pygame):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if (
                self.auto_button_rect is not None
                and self.auto_button_rect.collidepoint(mx, my)
                and not self.auto_solver_running
            ):
                self._start_auto_solver()
                return
            if self.follow_button_rect is not None and self.follow_button_rect.collidepoint(mx, my):
                self._toggle_camera_follow()
                return
            if self.target_mode_button_rect is not None and self.target_mode_button_rect.collidepoint(mx, my):
                self._toggle_target_mode()
                return
            for slider in self.sliders:
                if slider.rect.collidepoint(mx, my):
                    slider.dragging = True
                    slider.set_from_mouse(mx)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for slider in self.sliders:
                slider.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            mx, _ = event.pos
            for slider in self.sliders:
                if slider.dragging:
                    slider.set_from_mouse(mx)

    def _handle_keydown(self, key, pygame):
        if key == pygame.K_SPACE:
            self._launch_missile()
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_UP:
            self.launch_elevation_deg = min(89.0, self.launch_elevation_deg + 1.0)
        elif key == pygame.K_DOWN:
            self.launch_elevation_deg = max(1.0, self.launch_elevation_deg - 1.0)
        elif key == pygame.K_RIGHT:
            self.launch_speed = min(1500.0, self.launch_speed + 5.0)
        elif key == pygame.K_LEFT:
            self.launch_speed = max(1.0, self.launch_speed - 5.0)
        elif key == pygame.K_t:
            self.target_velocity_x += 5.0
        elif key == pygame.K_g:
            self.target_velocity_x -= 5.0
        elif key == pygame.K_q:
            self.launch_azimuth_deg -= 1.0
        elif key == pygame.K_e:
            self.launch_azimuth_deg += 1.0
        elif key == pygame.K_c:
            self._toggle_camera_follow()
        elif key == pygame.K_m:
            self._toggle_target_mode()
        elif key == pygame.K_v:
            self._reset_camera_view()

    def run(self):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is not installed. Install it in your venv with: ./venv/bin/pip install pygame") from exc

        pygame.init()
        pygame.display.set_caption("Ballistic Simulator - Interactive 3D Close-to-Life Mode")
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        clock = pygame.time.Clock()

        self._font_main = pygame.font.SysFont("consolas", 20)
        self._font_small = pygame.font.SysFont("consolas", 16)
        self._font_title = pygame.font.SysFont("consolas", 22)
        self._initialize_sliders(pygame)
        self._begin_camera_goals()

        while self.running:
            dt = clock.tick(60) / 1000.0
            self.current_mouse = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key, pygame)
                self._handle_slider_events(event, pygame)
                self._handle_mouse_camera(event, pygame)

            keys = pygame.key.get_pressed()
            self._handle_pressed_keys(keys, dt, pygame)
            self._apply_camera_follow(dt)
            self._update_camera_smooth(dt)

            self._apply_slider_values()
            self._update_physics(dt * self.time_scale)
            self._update_auto_solver()
            self._draw_scene(screen, pygame)
            pygame.display.flip()

        pygame.quit()
