import time
import numpy as np
import torch
import gym
from RL_method.DQN import DQN_Agent,ReplayBuffer,device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from scripts.arguments import add_arguments
from RL_method.utils import evaluate_policy,str2bool,all_evaluate_policy
from scripts.python_simulator.env import Env
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
argparser = argparse.ArgumentParser(description='CARLA CILQR')
add_arguments(argparser)
args = argparser.parse_args()

if __name__=="__main__":

    BriefEnvName = ['MyCar']
    Env_With_DW = [True] #DW: Die or Win
    args.env_with_dw = Env_With_DW[args.EnvIdex]
    env = Env(args)
    eval_env = Env(args)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_e_steps = env._max_episode_steps

    # Use DDQN or DQN
    if args.DDQN:
        algo_name = 'DDQN'
    else:
        algo_name = 'DQN'

    # Seed everything
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    args.Loadmodel=True
    args.ModelIdex=10000
    model = DQN_Agent(args)
    if args.Loadmodel: model.load(algo_name,BriefEnvName[args.EnvIdex],args.ModelIdex)
    score = all_evaluate_policy(eval_env,  True,model, 1)
    print(score)
    #1279，3085未进入终点
    #410碰撞2268
    #2015超车未进入

    #成功3088

    #410复杂环境碰撞
    #986速度体现
    #2268未停止，选择超车
    #3345未停止
    #2015超车未进入

