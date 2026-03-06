import numpy as np
from config import G


""" 
this function takes:
- v: initial speed 
- angle_degree: the degree at which the missile got launched 
- g: the value for gravity

returns position change accross time until the missile reachs ground 
and also time of flight and max horizontal range
"""
def trajectory(v: float, angle_degree: float, g: float = G):

  # calculate radian value of angle
  a = np.radians(angle_degree)

  # calculate time of flight 
  T = (2 * v * np.sin(a)) / g

  # create an array that will represent the x axis values for time
  t = np.linspace(0, T, 500)

  # compute horizontal position for each time value 
  x = v * np.cos(a) * t

  # compute vertical position for each time value 
  y = v * np.sin(a) * t - .5 * g * t**2

  # clip underground vertical position
  y = np.clip(y, 0, None)

  # compute the max horizontal range
  R = (v**2 * np.sin(2*a)) / g

  # compute maximum height the missile can reach
  H = (v**2 * (np.sin(a))**2) / (2 * g)

  return x, y, R, T, H
