import time
import numpy as np
import torch
import gym
from RL_method.DQN import DQN_Agent,ReplayBuffer,device
from torch.utils.tensorboard import SummaryWriter
import os, shutil,sys
from datetime import datetime
import argparse
from scripts.arguments import add_arguments
from RL_method.utils import evaluate_policy,str2bool
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
argparser = argparse.ArgumentParser(description='HCTRL')
add_arguments(argparser)
args = argparser.parse_args()

if __name__=="__main__":
    if args.use_ESC:
        BriefEnvName = ['ESC-DDQN-CILQR']
    else:
        BriefEnvName = ['DDQN-CILQR']
    Env_With_DW = [True] #DW: Die or Win
    args.env_with_dw = Env_With_DW[args.EnvIdex]
    env = Env(args)
    eval_env = Env(args)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_e_steps = env._max_episode_steps
    #Use DDQN or DQN
    if args.DDQN: algo_name = 'DDQN'
    else: algo_name = 'DQN'
    #Seed everything
    torch.manual_seed(args.seed)
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    # eval_env.seed(args.seed)
    eval_env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    print('Algorithm:', algo_name, '  Env:', BriefEnvName[args.EnvIdex], '  state_dim:', args.state_dim,
          '  action_dim:', args.action_dim, '  Random Seed:', args.seed, '  max_e_steps:', args.max_e_steps, '\n')
    if args.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(algo_name,BriefEnvName[args.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    model = DQN_Agent(args)
    if args.Loadmodel: model.load(algo_name,BriefEnvName[args.EnvIdex],args.ModelIdex)
    buffer = ReplayBuffer(args, max_size=int(1e6))

    if args.render:
        score = evaluate_policy(eval_env, model, True, 20)
        print('EnvName:', BriefEnvName[args.EnvIdex], 'seed:', args.seed, 'score:', score)

    else:
        total_steps = 0
        while total_steps < args.Max_train_steps:
            s, done, ep_r, steps = env.reset(), False, 0, 0
            while not done:
                steps += 1  #steps in current episode
                #e-greedy exploration
                if buffer.size < args.random_steps:
                    a = env.action_space.sample()
                else: a = model.select_action(s, deterministic=False)
                a = model.select_action(s, deterministic=False)
                s_prime, r, done, info = env.step(a)

                '''Avoid impacts caused by reaching max episode steps'''
                if done and steps != args.max_e_steps:
                    if args.EnvIdex == 0:
                        if r <= -100: r = -10
                    dw = True  # dw: dead and win
                else:
                    dw = False
                buffer.add(s, a, r, s_prime, dw)
                s = s_prime
                ep_r += r

                '''update if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= args.random_steps and total_steps % args.update_every == 0:
                    for j in range(args.update_every):
                        model.train(buffer)

                '''record & log'''

                total_steps += 1

                '''save model'''
                if (total_steps) % args.save_interval == 0:
                    model.save(algo_name, BriefEnvName[args.EnvIdex], total_steps)
    env.close()
