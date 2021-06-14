import mujoco_py
import numpy as np
import roboticstoolbox as rtb
import os
from controllers import Controllers
from mujoco_py import functions
import mujoco_py
from numpy.core.fromnumeric import resize
from trajectory_generator import tpoly
import matplotlib.pyplot as plt
from math import pi
import sys


mj_model = mujoco_py.load_model_from_path("/home/christian/Impedance_admittance_controller_in_MuJoCo/assets/full_kuka_all_joints.xml")
mj_data = mujoco_py.MjSim(mj_model)
# mj_simulation = mujoco_py.MjViewer(mj_data)
t = 0

""" pos, pos_two, pos_three = mj_data.data.sensordata[:3], mj_data.data.sensordata[3:6], mj_data.data.sensordata[6:9]
print(pos)
print(pos_two)
print(pos_three) """

""" while True:
    t += 1
    mj_data.step()
    mj_simulation.render()
    if t > 100 and os.getenv('Testing') is not None:
        break """
        
mj_simulation = mujoco_py.MjViewer(mj_data)

qi = np.zeros(7)
qf = np.array([0, 0, 0, 0, 0, 0.1, 0])
x = Controllers(mj_model, mj_data, qi, qf)
n = 2000
n_s = 2000
x.trajectory_generator(n, n_s)
# x.tau_actuator()
j = 0
while t < 10000:
    if j < n:
        x.tau_actuator(mj_data, j)
    t += 1
    j += 1
    #print(mj_data.data.qpos)
    #print(mj_data.data.qacc)
    mj_data.step()
    mj_simulation.render()
    
    if t > 100 and os.getenv('Testing') is not None:
        break

x.plot()
