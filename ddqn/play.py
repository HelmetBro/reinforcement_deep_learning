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
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
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
		file_path = 'models/' + model_file
		self.policy_net.load_state_dict(torch.load(file_path))
		self.policy_net.to(device)

	def choose_action(self, state):
		state = torch.FloatTensor(np.float32(state)).to(self.device).unsqueeze(0)
		q_value = self.policy_net(state).detach()
		action = q_value.max(1)[1].item()
		return action


def plot(frame_idx, rewards, losses):
	plt.figure(figsize=(20, 5))
	plt.subplot(131)
	plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
	plt.plot(rewards)
	plt.subplot(132)
	plt.title('loss')
	plt.plot(losses)
	plt.savefig('ddqn1_images/' + str(frame_idx) + '.jpg')
	plt.cla()
	plt.clf()
	plt.close('all')

def main():
	if len(sys.argv) != 2:
		print('Please input an environment to play')
		return
	if (sys.argv[1] == 'pong'):
		ENV_ID = "PongNoFrameskip-v4"
		model_file = "pong"
	elif (sys.argv[1] == 'space_invaders'):
		ENV_ID = "SpaceInvadersNoFrameskip-v4"
		model_file = "space_invaders"
	elif (sys.argv[1] == 'breakout'):
		ENV_ID = "BreakoutNoFrameskip-v4"
		model_file = "breakout"
	else:
		print('invalid game')
		return

	env = make_atari(ENV_ID)
	if (sys.argv[1] == 'pong'):
		env = wrap_deepmind(env)
		env = wrap_pytorch(env)
	else:
		env = wrap_deepmind(env, frame_stack=True, optimized=True)
	
	state = env.reset()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	ddqn = DDQN(env, device, model_file)

	num_frames = 1000000
	replay_initial = 10000

	losses = []
	all_rewards = []
	episode_reward = 0

	state_time = time.time()

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

	print(f'training took {time.time() - start_time} seconds')

if __name__ == "__main__":
	main()
