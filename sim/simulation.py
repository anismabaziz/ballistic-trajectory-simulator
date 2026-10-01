import math
import random
import threading

from .autosolve import DEFAULT_CANDIDATE_BUDGET, solve_launch
from .camera import apply_follow, handle_keyboard, handle_mouse, is_over_controls
from .drawing import draw_hud
from .ui import draw_control_panel, refresh_control_layout


def describe_auto_solution(result):
    """The status line the renderer shows for a solved launch, and its colour.

    The budget wording is the point of this function. A search that spent its
    whole grid returns the best launch solution available, and a search that ran
    out of budget returns the best it had looked at so far. Those are different
    claims about the same numbers, and the window has to make them different
    words or the search looks better than it is.
    """
    if not result.get("ok"):
        return "Auto solve failed", (255, 140, 140)

    speed = result["speed"]
    elevation = result["elevation"]
    azimuth = result["azimuth"]

    if result["hit"]:
        outcome = f"Auto launched hit: {speed:.1f}m/s, elev {elevation:.1f}, az {azimuth:.1f}"
        color = (120, 240, 150)
    else:
        outcome = (
            f"Auto launched best: {speed:.1f}m/s, elev {elevation:.1f}, "
            f"az {azimuth:.1f}, miss {result['miss_distance']:.1f}m"
        )
        color = (255, 210, 130)

    used = result["candidates_used"]
    if result["exhausted"]:
        return f"{outcome} - budget reached, best of {used}/{result['candidate_budget']} candidates", color
    return f"{outcome} - best of the search, {used} candidates", color


class _Projectile:
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
        self.miss_distance = float("inf")


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
        self.min_launch_elevation_deg = 1.0
        self.min_auto_solve_elevation_deg = 12.0
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
        self.projectiles = []
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
        # Auto Solve asks for the whole grid. The search runs on its own thread,
        # so a full sweep only costs the button's wait, and the wait buys an
        # answer that is the best of everything the search looked at rather than
        # a prefix of it. Lowering this makes the solve quicker and the answer
        # worse, and the status line will say so.
        self.auto_solve_candidate_budget = DEFAULT_CANDIDATE_BUDGET

        self.target_hit_timer = 0.0

        self.camera_follow_target = True
        self.camera_follow_offset_x = -220.0
        self.camera_follow_offset_y = 160.0
        self.camera_follow_offset_z = 0.0

        self.current_mouse = (0, 0)
        self.orbit_dragging = False
        self.pan_dragging = False
        self.last_mouse_pos = None

        self._init_environment_visuals()

    def _init_environment_visuals(self):
        rng = random.Random(7)
        self._sky_stars = []
        for _ in range(90):
            x = rng.randint(0, max(1, self.window_width - 1))
            y = rng.randint(0, max(1, int(self.window_height * 0.45)))
            b = rng.randint(120, 255)
            self._sky_stars.append((x, y, b))

    def reset(self):
        self.projectiles = []
        self.sim_time = 0.0
        self.target_hit_timer = 0.0

    def _initialize_sliders(self, pygame):
        self.sliders = [
            _Slider("Wind X", 0.0, 40.0, float(self.base_wind_x_vals[-1]), pygame.Rect(0, 0, 1, 1)),
            _Slider("Wind Z", -30.0, 30.0, 0.0, pygame.Rect(0, 0, 1, 1)),
            _Slider("Vert Wind", -15.0, 15.0, 0.0, pygame.Rect(0, 0, 1, 1)),
            _Slider("Air Density", 0.7, 1.5, self.base_rho, pygame.Rect(0, 0, 1, 1)),
            _Slider("Drag Coeff", 0.1, 1.2, self.base_drag, pygame.Rect(0, 0, 1, 1)),
            _Slider("Time Scale", 0.2, 4.0, self.time_scale, pygame.Rect(0, 0, 1, 1)),
        ]
        refresh_control_layout(self, pygame)

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

    def _start_auto_solver(self):
        if self.auto_solver_running:
            return
        self.auto_status_text = "Auto solving..."
        self.auto_status_color = (140, 200, 255)
        self.auto_solver_running = True
        self.auto_solver_result = None
        self.auto_solver_thread = threading.Thread(target=self._auto_solver_worker_simple, daemon=True)
        self.auto_solver_thread.start()

    def _auto_solver_worker_simple(self):
        snapshot = {
            "target_x_launch": float(self._target_position()[0]),
            "target_velocity_x": float(self.target_velocity_x),
            "target_radius": float(self.target_radius),
            "gravity": float(self.simulator.gravity),
            "mass": float(self.simulator.mass),
            "rho": float(self.simulator.rho),
            "drag_coefficient": float(self.simulator.drag_coefficient),
            "area": float(self.simulator.area),
            "latitude": float(self.simulator.latitude),
            "alt_levels": self.simulator.alt_levels,
            "wind_x_vals": self.simulator.wind_x_vals,
            "wind_z_vals": self.simulator.wind_z_vals,
            "wind_vertical_vals": self.simulator.wind_vertical_vals,
            "min_auto_elevation": float(self.min_auto_solve_elevation_deg),
            "candidate_budget": int(self.auto_solve_candidate_budget),
        }
        result = solve_launch(snapshot)
        self.auto_solver_result = result
        self.auto_solver_running = False

    def _apply_auto_solver_result(self, result):
        if result.get("ok"):
            self.launch_speed = min(2200.0, result["speed"])
            self.launch_elevation_deg = max(self.min_auto_solve_elevation_deg, result["elevation"])
            self.launch_azimuth_deg = result["azimuth"]
            self._launch_projectile()

        self.auto_status_text, self.auto_status_color = describe_auto_solution(result)

    def _update_auto_solver(self):
        if self.auto_solver_running:
            return
        if self.auto_solver_result is None:
            return

        result = self.auto_solver_result
        self.auto_solver_result = None
        self._apply_auto_solver_result(result)

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

    def _launch_projectile(self):
        elev = math.radians(self.launch_elevation_deg)
        azim = math.radians(self.launch_azimuth_deg)
        vx = self.launch_speed * math.cos(elev) * math.cos(azim)
        vy = self.launch_speed * math.sin(elev)
        vz = self.launch_speed * math.cos(elev) * math.sin(azim)
        self.projectiles.append(_Projectile(0.0, 0.0, 0.0, vx, vy, vz))

    def _target_position(self, t=None):
        tt = self.sim_time if t is None else float(t)
        x = self.target_x + self.target_velocity_x * tt
        return x, self.target_y, self.target_z

    def _update_physics(self, dt):
        self.sim_time += dt
        self.target_hit_timer = max(0.0, self.target_hit_timer - dt)
        tx, ty, tz = self._target_position()

        for projectile in self.projectiles:
            if projectile.alive:
                state = self.simulator.integrate_fixed_step(
                    [projectile.x, projectile.y, projectile.z, projectile.vx, projectile.vy, projectile.vz],
                    dt,
                    dt,
                    rule="rk4",
                )
                (
                    projectile.x,
                    projectile.y,
                    projectile.z,
                    projectile.vx,
                    projectile.vy,
                    projectile.vz,
                ) = state

                if projectile.y <= 0.0 and len(projectile.trail) > 3:
                    projectile.y = 0.0
                    projectile.alive = False

            dx = projectile.x - tx
            dy = projectile.y - ty
            dz = projectile.z - tz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            projectile.miss_distance = min(projectile.miss_distance, d)
            if d <= self.target_radius and projectile.alive:
                projectile.hit = True
                projectile.alive = False
                self.target_hit_timer = 1.2

            projectile.trail.append((projectile.x, projectile.y, projectile.z))
            if len(projectile.trail) > 1200:
                projectile.trail = projectile.trail[-1200:]

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
                depth = max((p1[2] + p2[2]) * 0.5, 1.0)
                shade = max(40, min(120, int(1600.0 / depth * 130.0)))
                pygame.draw.line(screen, (38, 55 + shade // 4, 42 + shade // 6), (p1[0], p1[1]), (p2[0], p2[1]), 1)

        for gz in range(z_min, z_max + 1, grid_spacing):
            p1 = self._project(x_min, 0.0, gz)
            p2 = self._project(x_max, 0.0, gz)
            if p1 and p2:
                depth = max((p1[2] + p2[2]) * 0.5, 1.0)
                shade = max(34, min(108, int(1500.0 / depth * 120.0)))
                pygame.draw.line(screen, (34, 48 + shade // 5, 40 + shade // 7), (p1[0], p1[1]), (p2[0], p2[1]), 1)

        for marker_x in range(0, x_max + 1, 1000):
            p = self._project(marker_x, 0.0, 0.0)
            if p:
                label = self._font_small.render(f"{marker_x}m", True, (186, 206, 180))
                screen.blit(label, (p[0] + 4, p[1] - 10))

        # Main axis highlight for orientation.
        axis_a = self._project(0.0, 0.0, 0.0)
        axis_b = self._project(x_max, 0.0, 0.0)
        if axis_a and axis_b:
            pygame.draw.line(screen, (132, 208, 152), (axis_a[0], axis_a[1]), (axis_b[0], axis_b[1]), 2)

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

        # Target ground shadow.
        shadow_r = max(3, int(radius_px * 1.35))
        pygame.draw.ellipse(screen, (16, 24, 18), (base[0] - shadow_r, base[1] - shadow_r // 2, shadow_r * 2, shadow_r))

        pygame.draw.circle(screen, fill_col, (body[0], body[1]), radius_px, 0)
        pygame.draw.circle(screen, edge_col, (body[0], body[1]), radius_px, 2)
        pygame.draw.circle(screen, (255, 255, 255), (body[0] - max(1, radius_px // 3), body[1] - max(1, radius_px // 3)), max(1, radius_px // 4))
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

    def _draw_projectile_shape(self, screen, pygame, projectile):
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

        center = (projectile.x, projectile.y, projectile.z)
        vel = (projectile.vx, projectile.vy, projectile.vz)
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

        body_color = (150, 160, 175) if not projectile.hit else (120, 210, 145)
        nose_color = (210, 215, 225) if not projectile.hit else (190, 255, 210)
        fin_color = (90, 110, 130) if not projectile.hit else (75, 170, 105)

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

    def _draw_projectiles(self, screen, pygame):
        for projectile in self.projectiles:
            if len(projectile.trail) > 1:
                pts = []
                for p in projectile.trail[-220:]:
                    pr = self._project(p[0], p[1], p[2])
                    if pr:
                        pts.append((pr[0], pr[1]))
                if len(pts) >= 2:
                    base_col = (120, 255, 165) if projectile.hit else (255, 205, 120)
                    pygame.draw.lines(screen, base_col, False, pts, 2)

                    # Soft glow trail points.
                    for i in range(0, len(pts), 5):
                        t = i / max(len(pts) - 1, 1)
                        r = int(base_col[0] * (0.55 + 0.45 * t))
                        g = int(base_col[1] * (0.55 + 0.45 * t))
                        b = int(base_col[2] * (0.55 + 0.45 * t))
                        pygame.draw.circle(screen, (r, g, b), pts[i], 2)

            self._draw_projectile_shape(screen, pygame, projectile)

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
        draw_control_panel(self, screen, pygame)

    def _draw_hud(self, screen, pygame):
        draw_hud(self, screen, pygame)

    def _is_over_slider_panel(self, pos):
        return is_over_controls(self, pos)

    def _apply_camera_follow(self, dt):
        apply_follow(self, dt)

    def _handle_mouse_camera(self, event, pygame):
        handle_mouse(self, event, pygame)

    def _draw_scene(self, screen, pygame):
        for row in range(self.window_height):
            t = row / max(self.window_height - 1, 1)
            r = int(8 * (1 - t) + 44 * t)
            g = int(16 * (1 - t) + 74 * t)
            b = int(30 * (1 - t) + 118 * t)
            pygame.draw.line(screen, (r, g, b), (0, row), (self.window_width, row))

        # Sun glow.
        sun_x = int(self.window_width * 0.16)
        sun_y = int(self.window_height * 0.18)
        for rad, alpha in ((82, 30), (52, 55), (24, 140)):
            surf = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 215, 145, alpha), (rad, rad), rad)
            screen.blit(surf, (sun_x - rad, sun_y - rad))

        # Stars.
        for sx, sy, b in self._sky_stars:
            screen.set_at((sx, sy), (b, b, min(255, b + 10)))

        self._draw_ground_grid(screen, pygame)
        self._draw_altitude_grid(screen, pygame)
        self._draw_target(screen, pygame)
        self._draw_projectiles(screen, pygame)
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
        handle_keyboard(self, keys, dt, pygame)

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
            self._launch_projectile()
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_UP:
            step = 5.0 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1.0
            self.launch_elevation_deg = min(89.0, self.launch_elevation_deg + step)
        elif key == pygame.K_DOWN:
            step = 5.0 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1.0
            self.launch_elevation_deg = max(self.min_launch_elevation_deg, self.launch_elevation_deg - step)
        elif key == pygame.K_RIGHT:
            self.launch_speed = min(2200.0, self.launch_speed + 5.0)
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
        import pygame

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
