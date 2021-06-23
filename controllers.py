import mujoco_py
import numpy as np
import os
from numpy.core.fromnumeric import resize
from trajectory_generator import tpoly
from scipy import integrate
import matplotlib.pyplot as plt
from math import pi
from mujoco_py import functions
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

mj_model = mujoco_py.load_model_from_path("/home/christian/Impedance_admittance_controller_in_MuJoCo/assets/full_kuka_all_joints.xml")
mj_data = mujoco_py.MjSim(mj_model)
t = 0
t_a_ = []
t_a_admittance = []
time = []
time_admittance = []
i = 0

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
        
        self.D_first_joint = 1
        self.D_second_joint = 1
        self.D_third_joint = 1
        self.D_fourth_joint = 1
        self.D_fifth_joint = 1
        self.D_sixth_joint = 1
        self.D_seventh_joint = 1
        
        self.K_first_joint = 1
        self.K_second_joint = 1
        self.K_third_joint = 1
        self.K_fourth_joint = 1
        self.K_fifth_joint = 1
        self.K_sixth_joint = 1
        self.K_seventh_joint = 1
        
        self.M_first_joint = 1
        self.M_second_joint = 1
        self.M_third_joint = 1
        self.M_fourth_joint = 1
        self.M_fifth_joint = 1
        self.M_sixth_joint = 1
        self.M_seventh_joint = 1
        
        self.t_ext_first_joint = 0
        self.t_ext_second_joint = 0
        self.t_ext_third_joint = 0
        self.t_ext_fourth_joint = 0
        self.t_ext_fifth_joint = 0
        self.t_ext_sixth_joint = 0.01
        self.t_ext_seventh_joint = 0
        
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
        
    def inertia_matrix(self, mj_data):
        self.I = np.zeros(7 * 7)
        mujoco_py.functions.mj_fullM(mj_data.model, self.I, mj_data.data.qM)
        self.I = np.reshape(self.I, (7, 7))
    
    def coriolis_matrix(self, mj_data):
        c = mj_data.data.qfrc_bias
        self.C = np.reshape(c, (7, 1))
    
    def tau_actuator_impedance_control(self, mj_data, j):
        """ 
            Iq_acc + C = t_a + t_e
            q_acc = I.inv(t_a + t_e - C) (1)
            
            t_e = Me_acc + Be_vel + Ke
            e = q - qd
            q_acc = M.inv(t_e + Mqd_acc - Be_vel - Ke) (2)
            
            (1) = (2):
            I.inv(t_a + t_e - C) = M.inv(t_e + Mqd_acc - Be_vel - Ke)
            t_a = (I(M.inv) - 1)t_e + Iqd_acc - I(M.inv)(Be_vel + Ke) + C
        """
        k = 5
        b = 2
        Kj = np.eye(7) * k
        Kj[0][0] = 300
        Kj[1][1] = 300
        Kj[2][2] = 200
        Kj[3][3] = 150
        Kj[4][4] = 10
        #Kj[5][5] = 1
        #Kj[6][6] = 10
        Bj = np.eye(7) * b
        Bj[1][1] = 10
        Bj[2][2] = 10
        Bj[3][3] = 10
        Bj[4][4] = 5
        Bj[5][5] = 5
        Bj[6][6] = 5
        
        
        
        self.inertia_matrix(mj_data)
        self.coriolis_matrix(mj_data)
        md = np.eye(7)
        # take the torque external reading from each joint 
        # external_torque = (self.I @ md) @ torque external reading - torque external reading
        # must add external_torque and + self.C
        t_a = ((self.I @ resize(self.acceleration_points[:, j], (7, 1))) -
            (Bj @ np.array([mj_data.data.qvel]).T) - (Kj @ np.array([mj_data.data.qpos]).T)
            + (Bj @ resize(self.velocity_points[:, j], (7, 1))) +
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
        
            
        """ print(t_a_a)
        print(mj_data.data.ctrl) """
        
    def plot(self):
        ax = plt.subplot(4, 1, 4)
        ax.plot(time, t_a_, "-o")
        plt.show()
    # ADMITTANCE CONTROL
    """ 
        The process aplied here used some math manipulations
        qd - q0 = a
        t_ext = Md(qd_dot_dot - q0_dot_dot) + Dd(qd_dot - q0_dot) + Kd(qd - q0)
        t_ext = Md(a_dot_dot) + Dd(a_dot) + Kd(a)
        
        - a - was passed to the functions of iteration to solve the ode
        After this, q0 was added to each a of each joint to obtain qd
        qd_admittance is the final result for this part where collums are
        the number of points in the trajetory of qd and rows are the DOF
    """ 
    def get_net_external_force(self, mj_data):
        self.net_external_force = np.array(mj_data.data.qfrc_inverse)
            
    def ode_second_order_solver_first_joint(self, Y, t):
        a_one_dot = Y[1]
        a_two_dot = ((self.net_external_force[0] - self.D_first_joint * Y[1] - self.K_first_joint * Y[0]) / self.M_first_joint)
        
        return [a_one_dot, a_two_dot]
    
    def ode_second_order_solver_second_joint(self, Y, t):
        a_one_dot = Y[1]
        a_two_dot = ((self.net_external_force[1] - self.D_second_joint * Y[1] - self.K_second_joint * Y[0]) / self.M_second_joint)
        
        return [a_one_dot, a_two_dot]
    
    def ode_second_order_solver_third_joint(self, Y, t):
        a_one_dot = Y[1]
        a_two_dot = ((self.net_external_force[2] - self.D_third_joint * Y[1] - self.K_third_joint * Y[0]) / self.M_third_joint)
        
        return [a_one_dot, a_two_dot]
    
    def ode_second_order_solver_fourth_joint(self, Y, t):
        a_one_dot = Y[1]
        a_two_dot = ((self.net_external_force[3] - self.D_fourth_joint * Y[1] - self.K_fourth_joint * Y[0]) / self.M_fourth_joint)
        
        return [a_one_dot, a_two_dot]
    
    def ode_second_order_solver_fifth_joint(self, Y, t):
        a_one_dot = Y[1]
        a_two_dot = ((self.net_external_force[4] - self.D_fifth_joint * Y[1] - self.K_fifth_joint * Y[0]) / self.M_fifth_joint)
        
        return [a_one_dot, a_two_dot]
    
    def ode_second_order_solver_sixth_joint(self, Y, t):
        a_one_dot = Y[1]
        a_two_dot = ((self.net_external_force[5] - self.D_sixth_joint * Y[1] - self.K_sixth_joint * Y[0]) / self.M_sixth_joint)
        
        return [a_one_dot, a_two_dot]
    
    def ode_second_order_solver_seventh_joint(self, Y, t):
        a_one_dot = Y[1]
        a_two_dot = ((self.net_external_force[6] - self.D_seventh_joint * Y[1] - self.K_seventh_joint * Y[0]) / self.M_seventh_joint)
        
        return [a_one_dot, a_two_dot]
           
    def output_admittance_control(self, t_stop):
        # Process to obtain qd
        t = np.arange(0, t_stop, 1)
        asol_one =  integrate.odeint(self.ode_second_order_solver_first_joint, [0, 0], t)
        asol_two =  integrate.odeint(self.ode_second_order_solver_second_joint, [0, 0], t)
        asol_three =  integrate.odeint(self.ode_second_order_solver_third_joint, [0, 0], t)
        asol_four =  integrate.odeint(self.ode_second_order_solver_fourth_joint, [0, 0], t)
        asol_five =  integrate.odeint(self.ode_second_order_solver_fifth_joint, [0, 0], t)
        asol_six =  integrate.odeint(self.ode_second_order_solver_sixth_joint, [0, 0], t)
        asol_seven =  integrate.odeint(self.ode_second_order_solver_seventh_joint, [0, 0], t)
        
        # self.qd_admittance is a matrix 7 x t_stop
        """ self.qd_admittance = [np.array(asol_one[:, 0]), np.array(asol_two[:, 0]),
               np.array(asol_three[:, 0]), np.array(asol_four[:, 0]),
               np.array(asol_five[:, 0]), np.array(asol_six[:, 0]),
               asol_seven[:, 0]] """
        
        self.qd_admittance = np.array([asol_one[:, 0], asol_two[:, 0],
               asol_three[:, 0], asol_four[:, 0],
               asol_five[:, 0], asol_six[:, 0],
               asol_seven[:, 0]])
        
        print(self.qd_admittance.shape)
        
        # sol[:] = np.add(sol, self.position_points[:, :len(t)])
        
        # Adding the q0 to obtain qd properly
        for j in range(len(t)):
            for z in range(7):
                self.qd_admittance[z][j] = self.qd_admittance[z][j] + self.position_points[z][j]
                
        self.qd_admittance[:] = resize(self.qd_admittance, (7, 10))

        """ print(sol)
        print(np.shape(sol))
        print(np.shape(self.position_points))
        print(np.shape(resize(asol_one[:, 0], (1, len(t))))) """
        
    def tau_admittance_control(self, mj_data, i):
        k_p = 5
        k_d = 2
        
        
        t_actuator = ((self.qd_admittance[:, i] - np.array([mj_data.data.qpos])) * k_p
                       - np.array([mj_data.data.qvel]) * k_d)
        
        t_a_admittance.append(t_actuator[0])
        time_admittance.append(i)
        t_a_admit = (resize(t_actuator, (1, 7)))[0, :6]
        mj_data.data.ctrl[:] = t_a_admit
        
        
        
        
        
        
    
if __name__ == '__main__':
    """ mj_simulation = mujoco_py.MjViewer(mj_data)
    
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
            break """
    
    qi = np.zeros(7)
    qf = np.array([0, 0, 0, 0, 0, pi/18, 0])
    x = Controllers(mj_model, mj_data, qi, qf)
    n = 2000
    n_s = 2000
    t_stop = 10
    x.trajectory_generator(n, n_s)
    x.output_admittance_control(t_stop)
    x.get_net_external_force(mj_data)
    i = 0
    t = 0
    mj_simulation = mujoco_py.MjViewer(mj_data)
    while t < 5000:
        if i < t_stop:
            x.get_net_external_force(mj_data)
            x.tau_admittance_control(mj_data, i)
        i += 1
        t += 1
        # print(mj_data.data.qpos)
        mj_data.step()
        mj_simulation.render()
        
        if t > 100 and os.getenv('Testing') is not None:
            break
    
    