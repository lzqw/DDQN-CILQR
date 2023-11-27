import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def process(s,s_net,args,device):
    obs=torch.zeros((s.shape[0],args.car_dim*args.obs_num+1+args.car_dim), dtype=torch.float).to(device)
    for index in range(s.shape[0]):
        s_per=s[index]
        s_per=s_per[s_per!=-1]
        s_per=s_per.reshape((-1, 4))
        s_per_ego=s_per[0]
        s_per_sur=s_per[1:]
        s_per_sur=s_net(s_per_sur).sum(axis=0)
        s_per=torch.hstack((s_per_ego,s_per_sur))
        obs[index]=s_per
    return obs
def build_net(layer_shape, activation, output_activation):
    '''build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape,args):
        super(Q_Net, self).__init__()
        layers = [args.car_dim*args.obs_num+1+args.car_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q

class S_Net(nn.Module):
    def __init__(self, args):
        super(S_Net, self).__init__()
        self.args=args
        self.hidden1=nn.Sequential(
                nn.Linear(in_features=args.car_dim,out_features=256,bias=True),
                nn.GELU())
        self.hidden2=nn.Sequential(
                nn.Linear(in_features=256,out_features=256,bias=True),
                nn.GELU())
        self.hidden3=nn.Sequential(
                nn.Linear(in_features=256,out_features=256,bias=True),
                nn.GELU())
        self.hidden4=nn.Sequential(
                nn.Linear(in_features=256,out_features=args.car_dim*args.obs_num+1,bias=True),
                nn.GELU())

    def forward(self, s):
        # s = s.reshape((-1, self.args.car_dim))
        s=self.hidden1(s)
        s=self.hidden2(s)
        s=self.hidden3(s)
        o=self.hidden4(s)
        return o


class DQN_Agent(object):
    def __init__(self, opt, ):
        self.opt=opt
        self.q_net = Q_Net(opt.state_dim, opt.action_dim, (opt.net_width, opt.net_width),opt).to(device)
        self.s_net=S_Net(opt).to(device)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr)
        self.s_net_optimizer = torch.optim.Adam(self.s_net.parameters(), lr=opt.lr)
        self.q_target = copy.deepcopy(self.q_net)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_target.parameters(): p.requires_grad = False
        self.env_with_dw = opt.env_with_dw
        self.gamma = opt.gamma
        self.tau = 0.0058
        self.batch_size = opt.batch_size
        self.exp_noise = opt.exp_noise
        self.action_dim = opt.action_dim
        self.DDQN = opt.DDQN

    def select_action(self, state, deterministic):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            state=process(state,self.s_net,self.opt,device)
            if deterministic:
                a = self.q_net(state).argmax().item()
            else:
                if np.random.rand() < self.exp_noise:
                    a = np.random.randint(0, self.action_dim)
                else:
                    a = self.q_net(state).argmax().item()
        return a

    def train(self, replay_buffer):
        s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)
        s_prime=process(s_prime,self.s_net,self.opt,device)
        s=process(s,self.s_net,self.opt,device)
        '''Compute the target Q value'''
        with torch.no_grad():
            if self.DDQN:
                argmax_a = self.q_net(s_prime).argmax(dim=1).unsqueeze(-1)
                max_q_prime = self.q_target(s_prime).gather(1, argmax_a)
            else:
                max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)

            '''Avoid impacts caused by reaching max episode steps'''
            if self.env_with_dw:
                target_Q = r + (1 - dw_mask) * self.gamma * max_q_prime  # dw: die or win
            else:
                target_Q = r + self.gamma * max_q_prime

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)
        self.q_net_optimizer.zero_grad()
        self.s_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()
        self.s_net_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, algo, EnvName, steps):
        torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.pth".format(algo, EnvName, steps))
        torch.save(self.s_net.state_dict(), "./model/{}_{}_{}_s.pth".format(algo, EnvName, steps))

    def load(self, algo, EnvName, steps):
        self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo, EnvName, steps)))
        self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo, EnvName, steps)))
        self.s_net.load_state_dict(torch.load("./model/{}_{}_{}_s.pth".format(algo, EnvName, steps)))


class ReplayBuffer(object):
    def __init__(self, args, max_size=int(1e6)):
        self.args=args
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        if self.args.use_esc==False:
            self.state = np.zeros((max_size, self.args.state_dim))
            self.next_state = np.zeros((max_size, self.args.state_dim))
        else:
            self.state = -1*np.ones((max_size, self.args.obs_num*4))
            self.next_state = -1*np.ones((max_size, self.args.obs_num*4))

        self.dw = np.zeros((max_size, 1))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.device = device

    def add(self, state, action, reward, next_state, dw):

        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        if self.args.use_esc==False:
            self.state[self.ptr] = state
            self.next_state[self.ptr] = next_state
        else:
            self.state[self.ptr,:len(state)] = state
            self.next_state[self.ptr,:len(next_state)] = next_state
        self.dw[self.ptr] = dw  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        with torch.no_grad():
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.Tensor(self.action[ind]).long().to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.dw[ind]).to(self.device)
            )

    def save(self):
        if not os.path.exists('buffer'): os.mkdir('buffer')
        scaller = np.array([self.max_size, self.ptr, self.size], dtype=np.uint32)
        np.save("buffer/scaller.npy", scaller)
        np.save("buffer/state.npy", self.state)
        np.save("buffer/action.npy", self.action)
        np.save("buffer/reward.npy", self.reward)
        np.save("buffer/next_state.npy", self.next_state)
        np.save("buffer/dw.npy", self.dw)

    def load(self):
        scaller = np.load("buffer/scaller.npy")

        self.max_size = scaller[0]
        self.ptr = scaller[1]
        self.size = scaller[2]

        self.state = np.load("buffer/state.npy")
        self.action = np.load("buffer/action.npy")
        self.reward = np.load("buffer/reward.npy")
        self.next_state = np.load("buffer/next_state.npy")
        self.dw = np.load("buffer/dw.npy")
