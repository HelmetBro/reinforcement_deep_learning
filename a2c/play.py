import argparse
import os
import numpy as np

import sys

#this is from the external lib
from a2c.atari_wrappers import make_atari, wrap_deepmind

from a2c.a2c import Model
from a2c.policy import Policy
import imageio
import time

#what game you want to run
def to_args_game(v):
  return v.lower() in ("-e", "-g", "--env", "--game")

#what version of the model you'd want to run
def to_version_game(v):
  return v.lower() in ("-v", "--version", "-i", "--iteration")

#location of models
def to_path_loc_model(v):
  return v.lower() in ("-m", "--models")

#location of gifs
def to_path_loc_gif(v):
  return v.lower() in ("-g", "--gifs")

def main(game='SpaceInvadersNoFrameskip-v4', version=0, model_location='.', gif_location='.'):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    save_path = os.path.join(model_location, str(game) + "-" + str(version) + '.model')

    #some of the wrapper stuff
    env = make_atari(game)
    env = wrap_deepmind(env, frame_stack=True, clip_rewards=False, episode_life=True)

    #played around with the params, the ones on google seemed to work the fastest (not best, i dont have infinite time)
    model = Model(
        policy=Policy, ob_space=env.observation_space,
        ac_space=env.action_space, nenvs=1, nsteps=5, nstack=1,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        lr=7e-4, alpha=0.99, epsilon=1e-5, total_timesteps=int(1e9))
    
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
            name = gif_location + str(int(time.time())) + '.gif'
            imageio.mimsave(name, renders, duration=1/30)
            
            #reset
            renders = []
            env.reset()