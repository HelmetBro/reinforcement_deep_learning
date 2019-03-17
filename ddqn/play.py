import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

from collections import deque
from pdb import set_trace
import matplotlib.pyplot as plt
import random
import math
import time
from ddqn.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import sys

class CnnDQN(nn.Module):
	def __init__(self, input_shape, num_actions):
		super(CnnDQN, self).__init__()

		self.input_shape = input_shape
		self.num_actions = num_actions

		self.conv_layer = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU()
		)

		self.fc = nn.Sequential(
			nn.Linear(self.feature_size(), 512),
			nn.ReLU(),
			nn.Linear(512, self.num_actions)
		)

	def forward(self, x):
		x = self.conv_layer(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

	def feature_size(self):
		return self.conv_layer(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class DDQN(object):
	def __init__(self, env, device, model_file):
		self.device = device
		self.state_shape = env.observation_space.shape
		self.num_actions = env.action_space.n
		self.policy_net = CnnDQN(self.state_shape, self.num_actions)
		file_path = 'ddqn/models/' + model_file
		self.policy_net.load_state_dict(torch.load(file_path))
		self.policy_net.to(device)

	def choose_action(self, state):
		state = torch.FloatTensor(np.float32(state)).to(self.device).unsqueeze(0)
		q_value = self.policy_net(state).detach()
		action = q_value.max(1)[1].item()
		return action


def create_environment(game_name):
	if (game_name == 'pong'):
		ENV_ID = "PongNoFrameskip-v4"
		model_file = game_name
	elif (game_name == 'space_invaders'):
		ENV_ID = "SpaceInvadersNoFrameskip-v4"
		model_file = game_name
	elif (game_name == 'breakout'):
		ENV_ID = "BreakoutNoFrameskip-v4"
		model_file = game_name
	else:
		print('invalid game')
		return

	env = make_atari(ENV_ID)
	if (game_name == 'pong'):
		env = wrap_deepmind(env)
		env = wrap_pytorch(env)
	else:
		env = wrap_deepmind(env, frame_stack=True, optimized=True)
	return env

def play(env, game_name, seconds_to_play=10):
	state = env.reset()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	ddqn = DDQN(env, device, game_name)

	num_frames = 1000000
	replay_initial = 10000

	losses = []
	all_rewards = []
	episode_reward = 0

	start_time = time.time()

	for i in range(1, num_frames + 1):
		action = ddqn.choose_action(state)

		state_, reward, done, _ = env.step(action)
		env.render()
		time.sleep(0.015)

		state = state_
		episode_reward += reward

		if done:
			state = env.reset()
			all_rewards.append(episode_reward)
			episode_reward = 0

		if time.time() - start_time >= seconds_to_play:
			env.close()
			return

	print(f'training took {time.time() - start_time} seconds')
