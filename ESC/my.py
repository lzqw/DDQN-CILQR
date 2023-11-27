import torch
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import os
import argparse
from scripts.arguments import add_arguments
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def build_net(layer_shape, activation, output_activation):
    '''build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)
class Q_Net(nn.Module):
    def __init__(self, args):
        super(Q_Net, self).__init__()
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
argparser = argparse.ArgumentParser(description='CARLA CILQR')
add_arguments(argparser)
args = argparser.parse_args()
Q=Q_Net(args)
a=torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,-1,-1,-1,-1],
               [1,2,3,4,5,6,7,8,9,10,11,12,-1,-1,-1,-1]], dtype=torch.float)
a1=torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,-1,-1,-1,-1], dtype=torch.float)
a2=torch.tensor([1,2,3,4], dtype=torch.float)

# a=a[a!=-1]
# print(a)
# a = a.reshape((a.shape[0],-1, 4))
# print(a)
def process(s,args):
    obs=torch.zeros((s.shape[0],args.car_dim*args.obs_num+1+args.car_dim), dtype=torch.float)
    for index in range(s.shape[0]):
        s_per=s[index]
        s_per=s_per[s_per!=-1]
        s_per=s_per.reshape((-1, 4))
        s_per_ego=s_per[0]
        s_per_sur=s_per[1:]
        s_per_sur=Q(s_per_sur).sum(axis=0)
        s_per=torch.hstack((s_per_ego,s_per_sur))
        obs[index]=s_per
    print(obs.shape)
    return obs

process(a,args)
# Q(a)