import numpy as np
import config as config


""" 
this function takes:
- v: initial speed 
- angle_degree: the degree at which the missile got launched 
- g: the value for gravity

returns position change accross time until the missile reachs ground 
and also time of flight and max horizontal range
"""
def trajectory(v: float, angle_degree: float, m: float  = config.MASS, g: float = config.G, dt: float = 0.01):

  # calculate radian value of angle
  a = np.radians(angle_degree)

  # calculate initial components of the velocity 
  vx = v * np.cos(a)
  vy = v * np.sin(a)

  # set up initial horizontal and vertical positions
  x, y, t = 0.0, 0.0, 0.0

  xs = []
  ys = []

  # max height
  H = 0

  while y >= 0:

    # calculate current velocity
    velocity = np.sqrt(vx**2 + vy**2)

    # calculate current drag force
    Fd = 0.5 * config.RHO * config.CD * config.AREA * velocity**2

    # calculate acceleration components 
    if velocity != 0:
      ax = -(Fd / m) * (vx / velocity)
      ay = -g - (Fd / m) * (vy / velocity)
    else:
      ax = 0
      ay = -g

    # update velocity 
    vx += ax * dt
    vy += ay * dt

    # update position based on velocity 
    x += vx * dt
    y += vy * dt

    # add positions
    xs.append(x)
    ys.append(y)

    # update time 
    t += dt

    # update max height 
    if y > H:
      H = y

  # max missile range
  R = x
  T = t

  return np.array(xs), np.array(ys), R, T, H
