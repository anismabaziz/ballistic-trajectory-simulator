def is_over_controls(sim, pos):
    mx, my = pos
    return sim.slider_panel_x <= mx <= sim.slider_panel_x + sim.slider_panel_width and 0 <= my <= sim.slider_panel_height


def apply_follow(sim, dt):
    if not sim.camera_follow_target:
        return
    tx, ty, tz = sim._target_position()
    lead_t = 0.8
    tx = tx + sim.target_velocity_x * lead_t

    desired_x = tx + sim.camera_follow_offset_x
    desired_y = max(40.0, ty + sim.camera_follow_offset_y)
    desired_z = tz + sim.camera_follow_offset_z

    k = min(1.0, dt * 4.5)
    sim._camera_target_x_goal += (desired_x - sim._camera_target_x_goal) * k
    sim._camera_target_y_goal += (desired_y - sim._camera_target_y_goal) * k
    sim._camera_target_z_goal += (desired_z - sim._camera_target_z_goal) * k


def handle_mouse(sim, event, pygame):
    if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1 and not is_over_controls(sim, event.pos):
            sim.orbit_dragging = True
            sim.last_mouse_pos = event.pos
        elif event.button == 3 and not is_over_controls(sim, event.pos):
            sim.pan_dragging = True
            sim.last_mouse_pos = event.pos
        elif event.button == 4:
            sim._camera_distance_goal = max(600.0, sim._camera_distance_goal * 0.92)
        elif event.button == 5:
            sim._camera_distance_goal = min(16000.0, sim._camera_distance_goal * 1.08)
    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            sim.orbit_dragging = False
        elif event.button == 3:
            sim.pan_dragging = False
    elif event.type == pygame.MOUSEMOTION and sim.last_mouse_pos is not None:
        dx = event.pos[0] - sim.last_mouse_pos[0]
        dy = event.pos[1] - sim.last_mouse_pos[1]
        sim.last_mouse_pos = event.pos

        if sim.orbit_dragging:
            sim._camera_yaw_goal += dx * 0.22
            sim._camera_pitch_goal = max(-8.0, min(85.0, sim._camera_pitch_goal - dy * 0.18))
        elif sim.pan_dragging:
            pan_scale = 1.8
            if sim.camera_follow_target:
                sim.camera_follow_offset_x -= dx * pan_scale
                sim.camera_follow_offset_y += dy * pan_scale
            else:
                sim._camera_target_x_goal -= dx * pan_scale
                sim._camera_target_y_goal += dy * pan_scale


def handle_keyboard(sim, keys, dt, pygame):
    speed_scale = 2.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0

    def axis_input(neg, pos):
        v = 0.0
        if neg:
            v -= 1.0
        if pos:
            v += 1.0
        return v

    def smooth(current, target, dt_local, rise=8.0, decay=9.0):
        if abs(target) > 1e-6:
            k = min(1.0, rise * dt_local)
            return current + (target - current) * k
        k = min(1.0, decay * dt_local)
        return current * (1.0 - k)

    yaw_input = axis_input(keys[pygame.K_j], keys[pygame.K_l])
    pitch_input = axis_input(keys[pygame.K_k], keys[pygame.K_i])
    zoom_input = axis_input(keys[pygame.K_o], keys[pygame.K_u])
    pan_x_input = axis_input(keys[pygame.K_a], keys[pygame.K_d])
    pan_z_input = axis_input(keys[pygame.K_w], keys[pygame.K_s])

    sim._camera_yaw_velocity = smooth(sim._camera_yaw_velocity, yaw_input * 70.0 * speed_scale, dt)
    sim._camera_pitch_velocity = smooth(sim._camera_pitch_velocity, pitch_input * 55.0 * speed_scale, dt)
    sim._camera_zoom_velocity = smooth(sim._camera_zoom_velocity, zoom_input * 1200.0 * speed_scale, dt)
    sim._camera_pan_x_velocity = smooth(sim._camera_pan_x_velocity, pan_x_input * 520.0 * speed_scale, dt)
    sim._camera_pan_z_velocity = smooth(sim._camera_pan_z_velocity, pan_z_input * 520.0 * speed_scale, dt)

    sim._camera_yaw_goal += sim._camera_yaw_velocity * dt
    sim._camera_pitch_goal = max(-10.0, min(80.0, sim._camera_pitch_goal + sim._camera_pitch_velocity * dt))
    sim._camera_distance_goal = max(600.0, min(16000.0, sim._camera_distance_goal + sim._camera_zoom_velocity * dt))
    if sim.camera_follow_target:
        sim.camera_follow_offset_x += sim._camera_pan_x_velocity * dt
        sim.camera_follow_offset_z += sim._camera_pan_z_velocity * dt
    else:
        sim._camera_target_x_goal += sim._camera_pan_x_velocity * dt
        sim._camera_target_z_goal += sim._camera_pan_z_velocity * dt
