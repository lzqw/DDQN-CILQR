import argparse
import logging
import os
import platform
import pdb
import math
import sys
import time
import random
import heapq
import numpy as np
from matplotlib import animation
from numpy import polyfit, poly1d
import matplotlib.pyplot as plt
import matplotlib.patches
from sklearn.preprocessing import normalize

import gym
from gym import spaces
# from gym.envs.classic_control import rendering
from numpy.polynomial.polynomial import Polynomial
# code for INTERACTION dataset
from INTERACTION_Sim.INSim import get_item_iterator, draw_map_without_lanelet, polygon_xy_from_motionstate, mapsize


class MotionState:
    def __init__(self, time_stamp_ms):
        assert isinstance(time_stamp_ms, int)
        self.time_stamp_ms = time_stamp_ms
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.psi_rad = None

    def __str__(self):
        return "MotionState: " + str(self.__dict__)


class Track:
    def __init__(self, id):
        # assert isinstance(id, int)
        self.track_id = id
        self.agent_type = None
        self.length = None
        self.width = None
        self.time_stamp_ms_first = None
        self.time_stamp_ms_last = None
        self.start_end = None
        self.motion_states = dict()
        self.road_states = dict()

    def __str__(self):
        string = "Track: track_id=" + str(self.track_id) + ", agent_type=" + str(self.agent_type) + \
                 ", length=" + str(self.length) + ", width=" + str(self.width) + \
                 ", time_stamp_ms_first=" + str(self.time_stamp_ms_first) + \
                 ", time_stamp_ms_last=" + str(self.time_stamp_ms_last) + \
                 "\n motion_states:"
        for key, value in sorted(self.motion_states.items()):
            string += "\n    " + str(key) + ": " + str(value)
        return string


try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except IndexError:
    print("Cannot add the common path {}".format(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.append(os.path.dirname(__file__))
from arguments import add_arguments
from PolyRect import PolyRect
from ilqr.iLQR import iLQR

# Add lambda functions
cos = lambda a: np.cos(a)
sin = lambda a: np.sin(a)
tan = lambda a: np.tan(a)

PI = math.pi
colors = ['r', 'g', 'b', 'k']


# Lanes defined at 4, 0, -4

class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, args):

        self.x_local_plan = None

        # env run set
        self.args = args
        self.K = args.K
        self.time_interval = 100
        self.done = False
        self.map_file_path = os.path.join(sys.path[1], 'maps', args.map_name, 'maps', args.map_name + '.osm')
        self.action_space = spaces.Discrete(args.low_controller_num)
        low_limits = np.array([0., 0., -self.args.max_speed, self.args.steer_angle_limits[0]])
        high_limits = np.array([1., 1., self.args.max_speed, self.args.steer_angle_limits[1]])

        if self.args.use_ESC:
            self.observation_space = spaces.Box(
                np.tile(low_limits, (1, self.args.ESC_max_agent + 1))[0],
                np.tile(high_limits, (1, self.args.ESC_max_agent + 1))[0]
            )
        else:
            self.observation_space = spaces.Box(
                np.tile(low_limits, (1, self.K + 1))[0]
                ,
                np.tile(high_limits, (1, self.K + 1))[0]
            )
        self._max_episode_steps = self.args.sim_time

        """init in track_init() function"""
        self.init_time = 0  # the init time of ego vehicle
        self.timestamp = 0  # time stamp of env(default interval is 100ms)
        self.track_dictionary = {}  # contain the track of map
        self.timestamp_min = 0
        self.timestamp_max = 0
        # state of ego car
        self.ego_id = 0  # the id of ego vehicle
        self.ego_car_x = []  # the global plan of ego car-x
        self.ego_car_y = []  # the global plan of ego car-x
        self.ego_car_length = 0
        self.ego_car_width = 0
        self.current_ego_state = []
        self.last_ego_states = []
        # map info
        self.end_state = []
        self.min_x, self.min_y, self.max_x, self.max_y = 0, 0, 0, 0

        """init in render() function"""
        self.fig = None
        self.axes = None
        self.text_dict = {}
        self.patches_dict = {}  # patch for vehicle
        self.local_plan_plot = None
        self.desired_plan_plot = None
        self.npc_plan_plot = None

        """init in reward"""
        self.end_distance = 2
        self.count = 0

        """init in get_k_npc() function"""
        self.navigation_agent = None
        self.id_k = []
        self.track_init()
        self.get_k_npc()

        self.create_ilqr_agent()

    def track_init(self):

        """Track loading"""
        self.track_dictionary = np.load('/home/lzqw/PycharmProjects/DDQN-CILQR/ESC/maps/DR_USA_Intersection_MA/track.npy',
                                        allow_pickle=True).item()
        self.timestamp_min = 1e9
        self.timestamp_max = 0
        if self.track_dictionary is not None:
            for key, track in get_item_iterator(self.track_dictionary):
                self.timestamp_min = min(self.timestamp_min, track.time_stamp_ms_first)
                self.timestamp_max = max(self.timestamp_max, track.time_stamp_ms_last)
        else:
            raise IOError("track dictionary is None")

        '''Random choosing ego car'''
        keys = list(self.track_dictionary.keys())
        # keys=[17,33,35,36,37,38,41,55,59,63,65,66,69,71,81,82,86,91,96,98,99,102,103,105,125,128,130,134,135,138,141,143,144,145,146,147,154,158,163]
        self.ego_id = random.sample(list(keys), 1)[0]
        # self.ego_id= 57
        # self.ego_id = self.args.ego_id
        ego_track = self.track_dictionary.get(self.ego_id)
        self.init_time = ego_track.time_stamp_ms_first
        self.timestamp = self.init_time

        '''Set the global plan of ego car'''
        self.ego_car_x = []
        self.ego_car_y = []
        for state in ego_track.motion_states.values():
            while state.x in self.ego_car_x or state.y in self.ego_car_y:
                state.x += 1e-3
                state.y += 1e-3
            if len(self.ego_car_x) > 0:
                if ((self.ego_car_x[-1] - state.x) ** 2 + (self.ego_car_y[-1] - state.y) ** 2) ** 0.5 >= 0.2:
                    self.ego_car_x.append(state.x)
                    self.ego_car_y.append(state.y)
            else:
                self.ego_car_x.append(state.x)
                self.ego_car_y.append(state.y)
        """---use the last "mylen" points to generate more point"""
        mylen = 4
        k = 0
        if mylen > len(self.ego_car_x):
            mylen = len(self.ego_car_x)
        for i in range(mylen - 1):
            k += self.ego_car_x[-(i + 1)] - self.ego_car_x[-(i + 2)]
        k = k / mylen
        x_next = [self.ego_car_x[-1] + k]
        for i in range(20):
            x_next.append(x_next[-1] + k)
        coeff = polyfit(self.ego_car_x[-3:], self.ego_car_y[-3:], 2)
        f = poly1d(coeff)
        y_next = f(x_next)
        self.ego_car_x.extend(x_next)
        self.ego_car_y.extend(y_next)

        self.ego_car_length = ego_track.length
        self.ego_car_width = ego_track.width
        state = list(ego_track.motion_states.values())[0]
        end_state = list(ego_track.motion_states.values())[-1]
        self.current_ego_state = np.array([state.x, state.y, np.linalg.norm((state.vx, state.vy)), state.psi_rad])
        self.last_ego_states = np.array([self.current_ego_state[0], self.current_ego_state[1]])
        self.end_state = np.array(
            [end_state.x, end_state.y, (end_state.vx ** 2 + end_state.vy ** 2) ** 0.5, end_state.psi_rad])

        self.min_x, self.min_y, self.max_x, self.max_y = mapsize(self.map_file_path, 0.0, 0.0)

    def get_k_npc(self):  # 3678
        dis_list, id_list, ms_list = [], [], []
        self.id_k, self.k_NPC_states = [], []

        for key, value in self.track_dictionary.items():
            if key != self.ego_id and value.time_stamp_ms_first <= self.timestamp <= value.time_stamp_ms_last:
                ms = value.motion_states[self.timestamp]
                dis_list.append(np.linalg.norm(np.array([ms.x, ms.y]) - self.current_ego_state[0:2]))
                id_list.append(key)
                ms_list.append((ms))

        min_index = [i for i, dis in enumerate(dis_list) if dis <= self.args.obs_dis] if self.args.use_ESC else map(
            dis_list.index, heapq.nsmallest(self.K, dis_list))

        for i in list(min_index):
            self.id_k.append(id_list[i])
            ms = ms_list[i]
            init_state = np.array([ms.x, ms.y, np.linalg.norm([ms.vx, ms.vy]), ms.psi_rad])
            NPCstates = [init_state]
            max_t = self.track_dictionary[id_list[i]].time_stamp_ms_last
            for j in range(self.args.horizon):
                t = self.timestamp + (j + 1) * 100
                ms_t = self.track_dictionary[id_list[i]].motion_states[min(t, max_t)]
                next_state = np.array([ms_t.x, ms_t.y, np.linalg.norm([ms_t.vx, ms_t.vy]), ms_t.psi_rad])
                NPCstates.append(next_state)
            self.k_NPC_states.append(np.array(NPCstates).T)

        self.k_NPC_states = np.array(self.k_NPC_states)
        self.num_vehicles = self.k_NPC_states.shape[0]
        if self.navigation_agent and self.navigation_agent.constraints.number_of_npc != self.num_vehicles:
            self.navigation_agent.constraints.set_k(self.num_vehicles)

    def get_state(self):
        if self.args.use_ESC == False:
            if self.k_NPC_states.shape[0] != 0:
                self.state = np.vstack((self.current_ego_state, self.k_NPC_states[:, :, 0]))
                self.state[:, 0] = (self.state[:, 0] - self.min_x) / (self.max_x - self.min_x)
                self.state[:, 1] = (self.state[:, 1] - self.min_y) / (self.max_y - self.min_y)
                if self.state.shape[0] < self.K + 1:
                    for i in range(self.K + 1 - self.state.shape[0]):
                        self.state = np.vstack((self.state, np.array([0., 0., 0., 0.])))
            else:
                self.state = self.current_ego_state
                for i in range(self.K):
                    self.state = np.vstack((self.state, np.array([0., 0., 0., 0.])))
        else:
            if self.k_NPC_states.shape[0] != 0:
                self.state = np.vstack((self.current_ego_state, self.k_NPC_states[:, :, 0]))
                self.state[:, 0] = (self.state[:, 0] - self.min_x) / (self.max_x - self.min_x)
                self.state[:, 1] = (self.state[:, 1] - self.min_y) / (self.max_y - self.min_y)
            else:
                self.state = self.current_ego_state
                self.state = np.vstack((self.state, np.array([0., 0., 0., 0.])))

    def check_collision(self):
        if self.k_NPC_states.shape[0] == 0:
            return False
        for i in range(self.num_vehicles - 1):
            d = np.linalg.norm(self.current_ego_state[0:2] - self.k_NPC_states[i, 0:2, 0])
            if d >= 3.0:
                return False
            else:
                return True

    def check_end(self):
        if np.linalg.norm(self.current_ego_state[0:2] - self.end_state[0:2]) <= self.end_distance:
            return True
        else:
            x = self.current_ego_state[0] - self.ego_car_x[-20:]
            y = self.current_ego_state[1] - self.ego_car_y[-20:]
            if np.min((x ** 2 + y ** 2) ** 0.5) <= self.end_distance:
                return True
            return False

    def plan_dis_reward(self):
        xy = np.dstack((self.ego_car_x, self.ego_car_y))[0]
        d = (xy - self.current_ego_state[0:2]) ** 2
        dis_reward = -np.min(np.sum(d, axis=1) ** 0.5) / 10
        return dis_reward

    def cal_reward(self):
        reward = 0
        reward += self.current_ego_state[2] / 10
        reward += self.plan_dis_reward()
        if self.check_collision():
            print('coll', self.ego_id)
            reward -= 100
            self.done = True
        else:
            reward += 0.1
        if self.control[0] < self.args.acc_limits[0] or self.control[0] > self.args.acc_limits[1]:
            reward -= 100
            print('acc', self.ego_id)
            self.done = True
        else:
            reward += 0.1
        if self.check_end():
            print('end', self.ego_id)
            reward += 10
            self.done = True
        if self.count >= self.args.sim_time:
            reward -= 100
            self.done = True
        return reward

    def low_level_controller_step(self):
        desired_path, local_plan, self.control = self.run_step_ilqr()
        self.current_ego_state = self.run_model_simulation(self.current_ego_state, self.control)
        self.last_ego_states = np.vstack((self.last_ego_states, self.current_ego_state[0:2]))
        self.x_local_plan = local_plan[:, 0]
        self.y_local_plan = local_plan[:, 1]
        self.x_desired_plan = desired_path[:, 0]
        self.y_desired_plan = desired_path[:, 1]

    def step(self, action):
        self.a = action
        self.navigation_agent.constraints.args.desired_speed = action
        reward = 0
        for i in range(self.args.p_times):
            if self.args.render:
                self.render()
            self.low_level_controller_step()
            reward += self.cal_reward()
            if self.done:
                break
        self.get_state()

        return self.state.flatten(), reward, self.done, None

    def render(self, mode='human', close=False):
        if close:
            pass
        if self.fig == None:
            self.fig, self.axes = plt.subplots(1, 1, figsize=(15, 15))
            lat_origin = 0.0
            lon_origin = 0.0
            # map_file_path=os.path.join(sys.path[1], 'maps', args.map_name, 'maps', args.map_name + '.osm')
            draw_map_without_lanelet(self.map_file_path, self.axes, lat_origin, lon_origin)
            self.axes.scatter(self.ego_car_x, self.ego_car_y, marker='.', linewidths=0.1, s=1)
            draw_circle = plt.Circle((self.end_state[0], self.end_state[1]), 3, fill=False)
            self.axes.add_artist(draw_circle)
            self.local_plan_plot, = plt.plot([], [], 'yo', ms=2)
            self.desired_plan_plot, = plt.plot([], [], 'bo', ms=1)
            self.npc_plan_plot, = plt.plot([], [], 'bo', ms=1)
        start_time = time.time()
        if -1 in self.text_dict:
            self.text_dict[-1].remove()
            self.text_dict.pop(-1)
        self.text_dict[-1] = self.axes.text(self.min_x + 10, self.max_y - 10, 'ego id:' + str(self.ego_id),
                                            horizontalalignment='center',
                                            zorder=30)
        for key, value in self.track_dictionary.items():
            if key != self.ego_id:
                if value.time_stamp_ms_first <= self.timestamp <= value.time_stamp_ms_last:
                    ms = value.motion_states[self.timestamp]
                    if key not in self.patches_dict:
                        width = value.width
                        length = value.length
                        if key in self.id_k:
                            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length),
                                                              closed=True,
                                                              zorder=20, color='g')
                        else:
                            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length),
                                                              closed=True,
                                                              zorder=20)
                        self.patches_dict[key] = rect
                        self.axes.add_patch(rect)
                        self.text_dict[key] = self.axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center',
                                                             zorder=30)
                    else:
                        width = value.width
                        length = value.length
                        self.patches_dict[key].remove()
                        self.patches_dict.pop(key)
                        self.text_dict[key].remove()
                        self.text_dict.pop(key)
                        if key in self.id_k:
                            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length),
                                                              closed=True,
                                                              zorder=20, color='g')
                        else:
                            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length),
                                                              closed=True,
                                                              zorder=20)
                        self.patches_dict[key] = rect
                        self.axes.add_patch(rect)
                        self.text_dict[key] = self.axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center',
                                                             zorder=30)
                else:
                    if key in self.patches_dict:
                        self.patches_dict[key].remove()
                        self.patches_dict.pop(key)
                        self.text_dict[key].remove()
                        self.text_dict.pop(key)

            else:
                if key not in self.patches_dict:
                    ms = value.motion_states[value.time_stamp_ms_first]
                    width = self.ego_car_width
                    length = self.ego_car_length
                    ms.x = self.current_ego_state[0]
                    ms.y = self.current_ego_state[1]
                    ms.psi_rad = self.current_ego_state[3]
                    rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length), closed=True,
                                                      zorder=20, color='r')
                    self.patches_dict[key] = rect
                    self.axes.add_patch(rect)
                    self.text_dict[key] = self.axes.text(ms.x, ms.y + 2, str(round(self.current_ego_state[2], 2)),
                                                         horizontalalignment='center',
                                                         zorder=30)
                else:
                    self.text_dict[key].remove()
                    self.text_dict.pop(key)
                    ms = value.motion_states[value.time_stamp_ms_first]
                    width = self.ego_car_width
                    length = self.ego_car_length
                    ms.x = self.current_ego_state[0]
                    ms.y = self.current_ego_state[1]
                    ms.psi_rad = self.current_ego_state[3]
                    self.patches_dict[key].set_xy(polygon_xy_from_motionstate(ms, width, length))
                    self.text_dict[key] = self.axes.text(ms.x, ms.y + 2,
                                                         str(round(self.current_ego_state[2], 2)) + '(' + str(self.a)
                                                         + ')', horizontalalignment='center',
                                                         zorder=30)
        end_time = time.time()
        plt.ion()
        if self.x_local_plan is not None:
            self.local_plan_plot.set_data(self.x_local_plan, self.y_local_plan)
            self.desired_plan_plot.set_data(self.x_desired_plan, self.y_desired_plan)
            xx, yy = [], []
            for i in range(self.k_NPC_states.shape[0]):
                xx.extend(list(self.k_NPC_states[i, 0, :]))
                yy.extend(list(self.k_NPC_states[i, 1, :]))
            self.npc_plan_plot.set_data(xx, yy)

        diff_time = end_time - start_time
        plt.pause(max(0.001, 100 / 1000. - diff_time))

    plt.ioff()

    def simulate_npc(self, init_state, control):
        self.NPC_states.append(init_state)
        control = np.hstack((control, np.zeros((2, self.args.horizon))))
        for i in range(control.shape[1] - 1):
            NPC_next_state = self.run_model_simulation(self.NPC_states[i], control[:, i])
            self.NPC_states.append(NPC_next_state)
        self.NPC_states = np.array(self.NPC_states).T

    def create_global_plan(self):
        self.plan_ilqr = []
        for i in range(len(self.ego_car_x)):
            self.plan_ilqr.append(np.array([self.ego_car_x[i], self.ego_car_y[i]]))
        self.plan_ilqr = np.array(self.plan_ilqr)

    def init_sim(self):  # C4
        return self.all_patches

    def get_ego_states(self):
        ego_states = np.array([[self.current_ego_state[0], self.current_ego_state[1], 0],
                               [self.current_ego_state[2], 0, 0],
                               [0, 0, self.current_ego_state[3]],
                               [0, 0, 0],
                               [0, 0, 0]])
        return ego_states

    def get_npc_bounding_box(self):
        return self.args.car_dims

    def create_ilqr_agent(self):
        self.create_global_plan()
        self.navigation_agent = iLQR(self.args, self.get_npc_bounding_box())
        self.navigation_agent.set_global_plan(self.plan_ilqr)

    def run_step_ilqr(self):
        assert self.navigation_agent != None, "Navigation Agent not initialized"
        self.get_k_npc()
        # print(self.k_NPC_states.shape)
        desired_path, local_plan, control = self.navigation_agent.run_step(self.get_ego_states(),
                                                                           self.k_NPC_states)
        self.count += 1
        self.timestamp = self.timestamp + self.time_interval

        return desired_path, local_plan, control[:, 0]

    def run_model_simulation(self, state, control):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        control[0] = np.clip(control[0], self.args.acc_limits[0], self.args.acc_limits[1])
        control[1] = np.clip(control[1], state[2] * tan(self.args.steer_angle_limits[0]) / self.args.wheelbase,
                             state[2] * tan(self.args.steer_angle_limits[1]) / self.args.wheelbase)

        Ts = self.args.timestep
        next_state = np.array([state[0] + cos(state[3]) * (state[2] * Ts + (control[0] * Ts ** 2) / 2),
                               state[1] + sin(state[3]) * (state[2] * Ts + (control[0] * Ts ** 2) / 2),
                               np.clip(state[2] + control[0] * Ts, 0.0, self.args.max_speed),
                               (state[3] + control[1] * Ts) % (2 * np.pi)])  # wrap angles between 0 and 2*pi
        # print("Next state {}".format(next_state))

        return next_state

    def reset(self):
        self.patches_dict = dict()
        self.text_dict = dict()
        self.npc_dict = dict()
        if self.args.render:
            plt.ion()
            plt.pause(0.5)
            plt.close()
            plt.ioff()

        self.fig = None
        self.axes = None
        self.done = False

        self.NPC_dict = {}
        self.patches = []
        self.x_local_plan = None
        self.y_local_plan = None
        self.x_desired_plan = None
        self.y_desired_plan = None
        self.track_init()
        self.create_ilqr_agent()
        self.get_k_npc()
        self.count = 0  # remember to reset

        self.get_state()
        return self.state.flatten()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='CARLA CILQR')
    add_arguments(argparser)
    args = argparser.parse_args()
    args.render=True
    pysim = Env(args)
    for i in range(2000):
        state, reward, done, _ = pysim.step(5)
        print(state)
        if done == True:
            pysim.reset()  # 986速度慢碰撞
