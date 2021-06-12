import mujoco_py
import numpy as np
import roboticstoolbox as rtb
import os
from controllers import Controllers
from mujoco_py import functions

mj_model = mujoco_py.load_model_from_path("/home/christian/Impedance_admittance_controller_in_MuJoCo/assets/full_kuka_all_joints.xml")
mj_data = mujoco_py.MjSim(mj_model)
# mj_simulation = mujoco_py.MjViewer(mj_data)
t = 0

""" pos, pos_two, pos_three = mj_data.data.sensordata[:3], mj_data.data.sensordata[3:6], mj_data.data.sensordata[6:9]
print(pos)
print(pos_two)
print(pos_three) """

p = Controllers(mj_model, mj_data)
jacobian = mj_data.data.body_jacp
jac = mj_data.data.body_jacr
#print(mj_data.data.geom_jacp)
#print(mj_data.data.get_site_jacp('peg_ft_site'))
#print(mj_data.data.get_site_jacp('peg_ft_site').shape)

# p.ee_jacobian()
p.trajectory_generator()

""" while True:
    t += 1
    mj_data.step()
    mj_simulation.render()
    if t > 100 and os.getenv('Testing') is not None:
        break """