import argparse
import os
import numpy as np

import sys

#this is from the external lib
from atari_wrappers import make_atari, wrap_deepmind

from a2c import Model
from policy import Policy
import imageio
import time

#this was pretty much copied and pasted from train.py then modified a bit to save gifs

def to_args_game(v):
  return v.lower() in ("-e", "-g", "--env", "--game")

def to_version_game(v):
  return v.lower() in ("-v", "--version")

def main():

    GAME = 'SpaceInvadersNoFrameskip-v4'
    VERSION = 0

    if len(sys.argv) > 1:
        if to_args_game(sys.argv[1]):
            GAME = sys.argv[2]
            save_path = os.path.join('models', GAME + '.model')

    #some of the wrapper stuff
    env = make_atari(GAME)
    env = wrap_deepmind(env, frame_stack=True, clip_rewards=False, episode_life=True)

    #played around with the params, the ones on google seemed to work the fastest (not best, i dont have infinite time)
    model = Model(
        policy=Policy, ob_space=env.observation_space,
        ac_space=env.action_space, nenvs=1, nsteps=5, nstack=1,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        lr=7e-4, alpha=0.99, epsilon=1e-5, total_timesteps=int(1e9))

    if len(sys.argv) > 2 and to_version_game(sys.argv[3]):
        VERSION = sys.argv[4]
        save_path = os.path.join('models', GAME + "-" + VERSION + '.model')
        
    else:
        save_path = os.path.join('models', GAME + '.model')
    
    model.load(save_path)

    obs = env.reset()

    #array to hold all the frames so we can save into .gifs
    renders = []

    while True:

        obs = np.expand_dims(obs.__array__(), axis=0)
        a, v, _ = model.step(obs)
        obs, reward, done, info = env.step(a)

        #getting the rendered frame, adding it to renders for gif creation
        #we're not training, so the overhead is acceptable
        renders.append(imageio.core.util.Image(env.render('rgb_array')))
        
        env.render()
        if done:
            
            #save recording as gif to gifs folder
            name = 'gifs/' + str(int(time.time())) + '.gif'
            imageio.mimsave(name, renders, duration=1/30)
            
            #reset
            renders = []
            env.reset()

main()
