"""
  Extended Kalman Filter (EKF) - Localaization sample
  Original author: Atsushi Sakai (@Atsushi_twi)
  Editor: minht57
"""
#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

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

show_animation = True

def calc_input():
  v = 1.0           # [m/s]
  yawRate = 0.1     # [rad/s]
  u = np.array([[v], [yawRate]])
  return u

def observation (xTrue, xd, u):
  xTrue = motion_model(xTrue, u)

  # Add noise to gps x-y
  z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

  # Add noise to input
  ud = u + INPUT_NOISE @ np.random.randn(2, 1)

  xd = motion_model(xd, ud)

  return xTrue, z, xd, ud

def motion_model(x, u):
  F = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]])

  B = np.array([[DT * math.cos(x[2, 0]), 0],
                [DT * math.sin(x[2, 0]), 0],
                [0.0, DT], 
                [1.0, 0.0]])

  x = F @ x + B @ u

  return x

def observation_model(x):
  H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
    ])

  z = H @ x

  return z

def jacobianF(x, u):
  """
  Jacobian of motion model

  motion_model
    x{t} = x{t-1} + v * dt * cos(yaw)
    y{t} = y{t-1} + v * dt * sin(yaw)

    yaw{t} = yaw{t-1} + omega * dt
    v{t} = v{t-1}

  so
    dx/d(yaw) = - v{t-1} * dt * sin(yaw)
    dx/dv = dt * cos(yaw)
    dy/d(yaw) = v * dt * cos(yaw)
    dy/dv = dt * sin(yaw)
  """
  yaw = x[2, 0]
  v = u[0, 0]
  jF = np.array([
    [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
    [0.0, 1.0,  DT * v * math.cos(yaw), DT * math.sin(yaw)],
    [0.0, 0.0,  1.0,                    0.0],
    [0.0, 0.0,  0.0,                    1.0]
    ])

  return jF

def jacobianH(x):
  # Jacobain of observation model
  jH = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
    ])

  return jH

def ekf_estimation(xEst, PEst, z, u):
  # Predict
  xPred = motion_model(xEst, u)
  jF = jacobianF(xPred, u)
  PPred = jF @ PEst @ jF.T + Q

  # Update
  jH = jacobianH(xPred)
  zPred = observation_model(xPred)
  y = z - zPred
  S = jH @ PPred @ jH.T + R
  K = PPred @ jH.T @ np.linalg.inv(S)
  xEst = xPred + K @ y
  PEst = (np.eye(len(xEst)) - K @ jH) @ PPred

  return xEst, PEst

def plot_covariance_ellipse(xEst, PEst):
  Pxy = PEst[0:2, 0:2]
  eigVal, eigVec = np.linalg.eig(Pxy)

  if eigVal[0] >= eigVal[1]:
    bigInd = 0
    smallInd = 1
  else:
    bigInd = 1
    smallInd = 0

  t = np.arange(0, 2 * math.pi + 0.1, 0.1)
  a = math.sqrt(eigVal[bigInd])
  b = math.sqrt(eigVal[smallInd])
  x = [a * math.cos(it) for it in t]
  y = [b * math.sin(it) for it in t]
  angle = math.atan2(eigVec[bigInd, 1], eigVec[bigInd, 0])
  R = np.array([
    [ math.cos(angle), math.sin(angle)],
    [-math.sin(angle), math.cos(angle)]
    ])
  fx = R @ np.array([x, y])
  px = np.array(fx[0, :] + xEst[0, 0]).flatten()
  py = np.array(fx[1, :] + xEst[1, 0]).flatten()
  plt.plot(px, py, "--r")

def main():
  print(__file__ + " start")

  time = 0.0

  # State vector [x, y, yaw, v]'
  xEst = np.zeros((4, 1))
  xTrue = np.zeros((4, 1))
  PEst = np.eye(4)

  # Dead reckoning
  xDR = np.zeros((4, 1))

  # History
  hxEst = xEst
  hxTrue = xTrue
  hxDR = xTrue
  hz = np.zeros((2, 1))

  while SIM_TIME >= time:
    time += DT
    u = calc_input()

    xTrue, z, xDR, ud = observation(xTrue, xDR, u)
    xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

    # Store data history
    hxEst = np.hstack((hxEst, xEst))
    hxDR = np.hstack((hxDR, xDR))
    hxTrue = np.hstack((hxTrue, xTrue))
    hz = np.hstack((hz, z))

    if show_animation:
      plt.cla()
      plt.plot(hz[0, :], hz[1, :], ".g")
      plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b")
      plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), "-k")
      plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r")
      plot_covariance_ellipse(xEst, PEst)
      plt.axis("equal")
      plt.grid(True)
      plt.pause(0.001)

if __name__ == '__main__':
  main()
