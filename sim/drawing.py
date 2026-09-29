import math


def draw_hud(sim, screen, pygame):
    panel_x = 12
    panel_y = 12
    panel_w = min(max(420, int(sim.window_width * 0.62)), sim.window_width - 24)
    panel_h = min(max(190, int(sim.window_height * 0.31)), sim.window_height - 24)

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((9, 14, 22, 182))
    screen.blit(panel, (panel_x, panel_y))

    # subtle border glow
    pygame.draw.rect(screen, (92, 132, 172), (panel_x, panel_y, panel_w, panel_h), width=1, border_radius=8)

    active = sum(1 for m in sim.projectiles if m.alive)
    hits = sum(1 for m in sim.projectiles if m.hit)
    closest = min((m.closest_distance for m in sim.projectiles), default=float("inf"))
    closest_text = "N/A" if math.isinf(closest) else f"{closest:.2f} m"

    info_lines = [
        f"Launch {sim.launch_speed:.1f} m/s | Elev {sim.launch_elevation_deg:.1f} deg | Azi {sim.launch_azimuth_deg:.1f} deg",
        f"Target x0 {sim.target_x:.0f} m | vx {sim.target_velocity_x:.1f} m/s | r {sim.target_radius:.1f} m",
        f"Target {'moving' if abs(sim.target_velocity_x) > 1e-9 else 'stationary'} | state {'HIT' if sim.target_hit_timer > 0.0 else 'ACTIVE'}",
        f"Projectiles total {len(sim.projectiles)} | active {active} | hits {hits}",
        f"Closest approach {closest_text} | Sim time {sim.sim_time:.2f} s",
    ]

    max_text_w = panel_w - 24
    y = panel_y + 10
    y = sim._draw_wrapped_text(screen, sim._font_main, "Interactive 3D Ballistic Simulator", (238, 243, 255), panel_x + 12, y, max_text_w, 22)

    for line in info_lines:
        if y > panel_y + panel_h - 44:
            break
        y = sim._draw_wrapped_text(screen, sim._font_small, line, (194, 210, 234), panel_x + 12, y, max_text_w, 18)

    controls = "SPACE launch, R reset, arrows speed/elev, Q/E azimuth, T/G target vx"
    controls_2 = "C follow cam, M target mode, V reset cam, SHIFT fast"
    footer_y = max(y + 4, panel_y + panel_h - 38)
    sim._draw_wrapped_text(screen, sim._font_small, controls, (148, 196, 170), panel_x + 12, footer_y, max_text_w, 16)
    sim._draw_wrapped_text(screen, sim._font_small, controls_2, (140, 182, 160), panel_x + 12, footer_y + 16, max_text_w, 16)
