import mujoco_py
import numpy as np
import os
from numpy.core.fromnumeric import resize
from trajectory_generator import tpoly
import matplotlib.pyplot as plt
from math import pi
from mujoco_py import functions
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

""" mj_model = mujoco_py.load_model_from_path("/home/christian/Impedance_admittance_controller_in_MuJoCo/assets/full_kuka_all_joints.xml")
mj_data = mujoco_py.MjSim(mj_model) """
t = 0
t_a_ = []
time = []

class Controllers:
    
    def __init__(self, mj_model, mj_data, qi, qf):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.force_sensor = mj_data.data.sensordata[:3]
        self.torque_sensor = mj_data.data.sensordata[3:6]
        self.velocity_sensor = mj_data.data.sensordata[6:9]
        self.position_sensor = mj_data.data.sensordata[9:12]
        self.joint_acceleration = np.array([mj_data.data.qacc])
        self.joint_velocity = np.array([mj_data.data.qvel])
        self.joint_position = np.array([mj_data.data.qpos])
        self.qi = qi
        self.qf = qf
        
    def ee_jacobian(self, mj_data):
        jacp = mj_data.data.get_site_jacp('peg_ft_site')
        jacr = mj_data.data.get_site_jacr('peg_ft_site')
        self.jac = np.resize(np.concatenate((jacp, jacr), axis=0), (6, 7))
    
    def trajectory_generator(self, n, n_s):
        trajectory_vector = [tpoly(self.qi[z], self.qf[z], np.linspace(0, n, n_s))
                             for z in range(7)]
        trajectory_points = [trajectory_vector[x].plot() for x in range(7)]
        self.position_points = [[trajectory_points[0][0][l] for l in range(len(trajectory_points[0][0]))],
                                [trajectory_points[1][0][l] for l in range(len(trajectory_points[1][0]))],
                                [trajectory_points[2][0][l] for l in range(len(trajectory_points[2][0]))],
                                [trajectory_points[3][0][l] for l in range(len(trajectory_points[3][0]))],
                                [trajectory_points[4][0][l] for l in range(len(trajectory_points[4][0]))],
                                [trajectory_points[5][0][l] for l in range(len(trajectory_points[5][0]))],
                                [trajectory_points[6][0][l] for l in range(len(trajectory_points[6][0]))]]
        self.position_points = resize(self.position_points, (len(self.position_points),len(self.position_points[0])))
        self.velocity_points = [[trajectory_points[0][1][l] for l in range(len(trajectory_points[0][1]))],
                                [trajectory_points[1][1][l] for l in range(len(trajectory_points[1][1]))],
                                [trajectory_points[2][1][l] for l in range(len(trajectory_points[2][1]))],
                                [trajectory_points[3][1][l] for l in range(len(trajectory_points[3][1]))],
                                [trajectory_points[4][1][l] for l in range(len(trajectory_points[4][1]))],
                                [trajectory_points[5][1][l] for l in range(len(trajectory_points[5][1]))],
                                [trajectory_points[6][1][l] for l in range(len(trajectory_points[6][1]))]]
        self.velocity_points = resize(self.velocity_points, (len(self.velocity_points),len(self.velocity_points[0])))
        self.acceleration_points = [[trajectory_points[0][2][l] for l in range(len(trajectory_points[0][2]))],
                                [trajectory_points[1][2][l] for l in range(len(trajectory_points[1][2]))],
                                [trajectory_points[2][2][l] for l in range(len(trajectory_points[2][2]))],
                                [trajectory_points[3][2][l] for l in range(len(trajectory_points[3][2]))],
                                [trajectory_points[4][2][l] for l in range(len(trajectory_points[4][2]))],
                                [trajectory_points[5][2][l] for l in range(len(trajectory_points[5][2]))],
                                [trajectory_points[6][2][l] for l in range(len(trajectory_points[6][2]))]]
        self.acceleration_points = resize(self.acceleration_points, (len(self.acceleration_points),len(self.acceleration_points[0])))
        #print(self.velocity_points)
        #print(resize(self.velocity_points[:, 0], (7, 1)))
        #print(len(self.velocity_points[0]))
        #print(self.position_points)
        #print(resize(self.position_points[:, 0], (7, 1)))
        
    def inertia_matrix(self, mj_data):
        self.I = np.zeros(7 * 7)
        mujoco_py.functions.mj_fullM(mj_data.model, self.I, mj_data.data.qM)
        self.I = np.reshape(self.I, (7, 7))
    
    def coriolis_matrix(self, mj_data):
        c = mj_data.data.qfrc_bias
        self.C = np.reshape(c, (7, 1))
    
    def tau_actuator(self, mj_data, j):
        k = 5
        b = 2
        Kj = np.eye(7) * k
        Kj[0][0] = 100
        Kj[1][1] = 50
        Kj[2][2] = 40
        Kj[3][3] = 10
        Kj[4][4] = 10
        #Kj[5][5] = 1
        #Kj[6][6] = 10
        Bj = np.eye(7) * b
        """ Bj[1][1] = 5
        Bj[2][2] = 5
        Bj[3][3] = 5
        Bj[4][4] = 5
        Bj[5][5] = 5
        Bj[6][6] = 5 """
        
        
        
        self.inertia_matrix(mj_data)
        self.coriolis_matrix(mj_data)
        """ t_a = ((self.I @ np.array([mj_data.data.qacc]).T) -
            (Bj @ np.array([mj_data.data.qvel]).T) - (Kj @ np.array([mj_data.data.qpos]).T) +
            self.C + (Bj @ resize(self.velocity_points[:, j], (7, 1))) +
            (Kj @ resize(self.position_points[:, j], (7, 1)))) """
        """ t_a = ((self.I @ np.array([mj_data.data.qacc]).T) +
            (Bj @ np.array([mj_data.data.qvel]).T) + (Kj @ np.array([mj_data.data.qpos]).T) +
            self.C - (Bj @ np.zeros(7).T) -
            (Kj @ self.qf.T)) """
        t_a = ((self.I @ resize(self.acceleration_points[:, j], (7, 1))) -
            (Bj @ np.array([mj_data.data.qvel]).T) - (Kj @ np.array([mj_data.data.qpos]).T) +
            self.C + (Bj @ resize(self.velocity_points[:, j], (7, 1))) +
            (Kj @ resize(self.position_points[:, j], (7, 1))))
        # print(np.array([self.position_points[:, j]]).shape)
        """ print("")
        print("Inertia")
        print(self.I @ np.array([mj_data.data.qacc]).T)
        print("")
        print("bj velocity")
        print(Bj @ np.array([mj_data.data.qvel]).T)
        print("")
        print("kj position")
        print(Kj @ np.array([mj_data.data.qpos]).T)
        print("")
        print("coriolis")
        print(self.C)
        print("")
        print("bj desired velocity")
        print(Bj @ resize(self.velocity_points[:, j], (7, 1)))
        print("")
        print("kj desired position")
        print(Kj @ resize(self.position_points[:, j], (7, 1))) """
                    
        
        t_a_.append(t_a[0])
        time.append(j)
        t_a_a = (resize(t_a, (1, 7)))[0, :6]
        mj_data.data.ctrl[:] = t_a_a
        
        """ self.inertia_matrix()
        self.coriolis_matrix()
        t_a = ((self.I @ np.array([mj_data.data.qacc]).T) +
            (Bj @ np.array([mj_data.data.qvel]).T) + (Kj @ np.array([mj_data.data.qpos]).T) +
            self.C - (Bj @ resize(self.velocity_points[:, j], (7, 1))) -
            (Kj @ np.reshape(self.qf, (7, 1))))
        print(t_a)
        t_a_.append(t_a[5])
        time.append(j)
        t_a_a = (resize(t_a, (1, 7)))[0, :6]
        mj_data.data.ctrl[:] = t_a_a """
            
        """ print(t_a_a)
        print(mj_data.data.ctrl) """
        
    def plot(self):
        ax = plt.subplot(4, 1, 4)
        ax.plot(time, t_a_, "-o")
        plt.show()
        
        
    
if __name__ == '__main__':
    mj_simulation = mujoco_py.MjViewer(mj_data)
    
    qi = np.zeros(7)
    qf = np.array([0, 0, 0, 0, 0, pi/18, 0])
    x = Controllers(mj_model, mj_data, qi, qf)
    n = 4
    x.trajectory_generator(n)
    # x.tau_actuator()
    j = 0
    while t < 5000:
        if j < n:
            x.tau_actuator(mj_data, j)
        t += 1
        j += 1
        # print(mj_data.data.qpos)
        mj_data.step()
        mj_simulation.render()
        
        if t > 100 and os.getenv('Testing') is not None:
            break
    