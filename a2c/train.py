from subproc_vec_env import SubprocVecEnv
from atari_wrappers import make_atari, wrap_deepmind

from policy import Policy
from a2c import learn

import os

import sys

import gym
import argparse

DEFAULT_GAME = 'SpaceInvadersNoFrameskip-v4'

def to_args_interval(v):
  return v.lower() in ("-s", "-i", "--interval_saves", "--intervals", "--saves")

def to_args_game(v):
  return v.lower() in ("-e", "-g", "--env", "--game")

def train(env_id, num_timesteps, num_cpu):

    GAME = env_id
    if len(sys.argv) > 1 and to_args_game(sys.argv[1]):
        GAME = sys.argv[2]

    def make_env(rank):
        def _thunk(): #i'd like to get rid of this but google says otherwise 
            env = make_atari(GAME)
            return wrap_deepmind(env)
        return _thunk

    #depending on the number of enviroments, open a virtual one using the wrapper
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    #option to save at each checkpoint of interval and specify interval length
    INT_SAVES = False
    INTERVAL = 1000
    
    if len(sys.argv) > 3 and to_args_interval(sys.argv[3]):
    
        INTERVAL = int(sys.argv[4])
        INT_SAVES = True
        learn(Policy, env, num_timesteps, interval_saves=INT_SAVES, INTERVAL=INTERVAL)

    else:
        learn(Policy, env, num_timesteps)

    env.close()

#i have an 8 core cpu with hyperthreading, so 8*2 threads in total for 16 environments
#super large timestep, just press ctrl+c to finish
train(DEFAULT_GAME, int(1e9), num_cpu=8*2)