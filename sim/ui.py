def refresh_control_layout(sim, pygame):
    panel_w = min(340, max(270, int(sim.window_width * 0.28)))
    panel_h = sim.window_height
    panel_x = sim.window_width - panel_w

    sim.slider_panel_width = panel_w
    sim.slider_panel_height = panel_h
    sim.slider_panel_x = panel_x

    slider_x = panel_x + 86
    slider_w = panel_w - 98
    slider_h = 14
    top = 64
    spacing = 40

    for idx, slider in enumerate(sim.sliders):
        slider.rect = pygame.Rect(slider_x, top + idx * spacing, slider_w, slider_h)

    buttons_top = sim.sliders[-1].rect.bottom + 20 if sim.sliders else 280
    half_w = (panel_w - 30) // 2
    sim.auto_button_rect = pygame.Rect(panel_x + 12, buttons_top, half_w, 30)
    sim.follow_button_rect = pygame.Rect(panel_x + 18 + half_w, buttons_top, half_w, 30)
    sim.target_mode_button_rect = pygame.Rect(panel_x + 12, buttons_top + 38, panel_w - 24, 30)


def draw_control_panel(sim, screen, pygame):
    refresh_control_layout(sim, pygame)
    panel_w = sim.slider_panel_width
    panel_h = sim.slider_panel_height

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((16, 22, 30, 206))
    screen.blit(panel, (sim.slider_panel_x, 0))

    title = sim._font_title.render("Environment Controls", True, (220, 230, 250))
    screen.blit(title, (sim.slider_panel_x + 12, 16))

    for slider in sim.sliders:
        y = slider.rect.y - 18
        lbl = sim._font_small.render(f"{slider.name}: {slider.value:.2f}", True, (176, 208, 255))
        screen.blit(lbl, (sim.slider_panel_x + 14, y))

        pygame.draw.rect(screen, (68, 84, 104), slider.rect, border_radius=6)
        fill_rect = pygame.Rect(slider.rect.x, slider.rect.y, int(slider.rect.width * slider.normalized()), slider.rect.height)
        pygame.draw.rect(screen, (108, 178, 255), fill_rect, border_radius=6)
        handle_x = slider.rect.x + int(slider.rect.width * slider.normalized())
        pygame.draw.circle(screen, (235, 242, 255), (handle_x, slider.rect.y + slider.rect.height // 2), 7)

    if sim.auto_button_rect is not None:
        button_color = (62, 132, 210) if not sim.auto_solver_running else (95, 95, 115)
        pygame.draw.rect(screen, button_color, sim.auto_button_rect, border_radius=8)
        pygame.draw.rect(screen, (110, 170, 235), sim.auto_button_rect, width=2, border_radius=8)
        button_name = "Auto Solve" if not sim.auto_solver_running else "Solving..."
        button_label = sim._font_small.render(button_name, True, (240, 246, 255))
        screen.blit(button_label, (sim.auto_button_rect.x + 14, sim.auto_button_rect.y + 7))

    if sim.follow_button_rect is not None:
        follow_color = (68, 150, 110) if sim.camera_follow_target else (110, 88, 88)
        pygame.draw.rect(screen, follow_color, sim.follow_button_rect, border_radius=8)
        pygame.draw.rect(screen, (190, 220, 205), sim.follow_button_rect, width=2, border_radius=8)
        follow_text = "Cam Follow: ON" if sim.camera_follow_target else "Cam Follow: OFF"
        screen.blit(sim._font_small.render(follow_text, True, (245, 248, 255)), (sim.follow_button_rect.x + 8, sim.follow_button_rect.y + 7))

    if sim.target_mode_button_rect is not None:
        moving = abs(sim.target_velocity_x) > 1e-9
        mode_color = (72, 138, 205) if moving else (108, 108, 120)
        pygame.draw.rect(screen, mode_color, sim.target_mode_button_rect, border_radius=8)
        pygame.draw.rect(screen, (176, 208, 246), sim.target_mode_button_rect, width=2, border_radius=8)
        mode_text = f"Target: {'Moving' if moving else 'Stationary'} (toggle)"
        screen.blit(sim._font_small.render(mode_text, True, (242, 247, 255)), (sim.target_mode_button_rect.x + 12, sim.target_mode_button_rect.y + 7))

    if sim.auto_status_text:
        anchor_y = sim.target_mode_button_rect.bottom + 10 if sim.target_mode_button_rect is not None else 300
        status_end_y = sim._draw_wrapped_text(
            screen,
            sim._font_small,
            sim.auto_status_text,
            sim.auto_status_color,
            sim.slider_panel_x + 14,
            anchor_y,
            panel_w - 28,
            16,
        )
    else:
        status_end_y = sim.target_mode_button_rect.bottom + 8 if sim.target_mode_button_rect is not None else 320

    help_lines = [
        "Keyboard camera:",
        "  I/K pitch, J/L yaw, U/O zoom, W/A/S/D pan",
        "Launch controls:",
        "  SPACE launch, R reset, arrows speed/elev, Q/E azimuth",
        "Modes:",
        "  C follow cam, M target mode, V cam reset",
        "Mouse:",
        "  Left drag orbit, right drag pan, wheel zoom",
    ]

    y = max((sim.target_mode_button_rect.bottom + 36) if sim.target_mode_button_rect is not None else 300, status_end_y + 8)
    max_text_width = panel_w - 28
    for line in help_lines:
        y = sim._draw_wrapped_text(
            screen,
            sim._font_small,
            line,
            (160, 190, 175),
            sim.slider_panel_x + 14,
            y,
            max_text_width,
            16,
        )
