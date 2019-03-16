from subproc_vec_env import SubprocVecEnv
from atari_wrappers import make_atari, wrap_deepmind

from policy import Policy
from a2c import learn

import os

import sys

import gym
import argparse

DEFAULT_GAME = 'SpaceInvadersNoFrameskip-v4'

def to_args_game(v):
  return v.lower() in ("-e", "-g", "--env", "--game")

def to_args_interval(v):
  return v.lower() in ("-i", "--intervals")

def to_args_save_loc(v):
  return v.lower() in ("-s", "--save", "-p", "--path")

def train(env_id, num_timesteps, num_cpu):

    GAME = env_id

    if len(sys.argv) > 1:
        GAME = sys.argv[2]
        INTERVAL = int(sys.argv[4])
        PATH = sys.argv[6]
        INT_SAVES = True


    def make_env(rank):
        def _thunk(): #i'd like to get rid of this but google says otherwise 
            env = make_atari(GAME)
            return wrap_deepmind(env)
        return _thunk

    #depending on the number of enviroments, open a virtual one using the wrapper
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    #option to save at each checkpoint of interval and specify interval length
    if len(sys.argv) > 1:
        learn(Policy, env, num_timesteps, interval_saves=INT_SAVES, INTERVAL=INTERVAL, PATH=PATH)
    else:
        learn(Policy, env, num_timesteps)

    env.close()

#i have an 8 core cpu with hyperthreading, so 8*2 threads in total for 16 environments
#super large timestep, just press ctrl+c to finish
train(DEFAULT_GAME, int(1e9), num_cpu=8*2)