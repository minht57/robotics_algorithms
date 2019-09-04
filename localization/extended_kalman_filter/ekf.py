"""
  Extended Kalman Filter (EKF) - Localaization sample
  Original author: Atsushi Sakai (@Atsushi_twi)
  Editor: minht57
"""

import math
import numpy as numpy
import matplotlib.pyplot as pyplot

# Covariance for EKF simulation
Q = np.diag([
  0.1,              # variance of location on x-axis
  0.1,              # variance of location on y-axis
  np.deg2rad(1.0),  # variance of yaw angle
  1.0               # variance of velocity
  ]) ** 2

# Observation x, y position covariance
R = np.diag([1.0, 1.0]) ** 2

# Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1            # time tick [s]

SIM_TIME = 50.0     # simulation time [s]

def calc_input():
  v = 1.0           # [m/s]
  yawRate = 0.1     # [rad/s]
  u = np.array([[v], [yawRate]])
  return u

def observation (xTrue, xd, u):
  xTrue = motion_model(xTrue, u)

  # Add noise to gps x-y
  z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

  xd = motion_model(xd, ud)

  return xTrue, z, xd, ud

def motion_model(x, u):
  F = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]])

  B = np.array([[DT * math.cos(x[2, 0], 0)]])
