# here is the DDQN that is trained on Atari games
# Some of the code from here is used as a reference: https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb

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
import pickle
import sys

# pong actions
# 0 - dont move
# 2/4 - move up
# 3/5 - move down

#ENV_ID = "SpaceInvadersNoFrameskip-v4"
#ENV_ID = "BreakoutNoFrameskip-v4"

ENV_ID = "PongNoFrameskip-v4"

BATCH_SIZE = 32
LR = 0.0001
# Used to calculate epsilon for epsilon greedy action selection
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 30000
# Future rewards decay
GAMMA = 0.99
# Train every nth frame
TRAINING_FREQUENCY = 1
# Update target network weights every [TARGET_REPLACE_ITER] frames
TARGET_REPLACE_ITER = 1000
# Total number of frames to train
NUM_FRAMES = 2000000
# Capacity for replay memory
MEMORY_CAPACITY = 100000
# Minimum transitions needed to store in replay memory before training is started
REPLAY_INITIAL = 10000
# Save Model and plot analytics every [ANALYTICS_FREQUENCY] frames
ANALYTICS_FREQUENCY = 10000
# Backup Replay Buffer Every [REPLAY_BUFFER_BACKUP_FREQUENCY] frames
#REPLAY_BUFFER_BACKUP_FREQUENCY = 100000

# Uses deque to store the last [MEMORY_CAPCITY] state transitions to randomly sample from during training.


class ReplayBuffer(object):
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		state = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)

		self.buffer.append((state, action, reward, next_state, done))

	def sample(self):
		state, action, reward, next_state, done = zip(
			*random.sample(self.buffer, BATCH_SIZE))

		return np.concatenate(state), action, reward, np.concatenate(next_state), done

	def __len__(self):
		return len(self.buffer)

# Gives you the current Decayed Epsilon amount based off the current frame (exponential decay)


def epsilon_by_frame(current_frame):
	return EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1. * current_frame / EPSILON_DECAY)

# Structure of the NN for both the policy and target network


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

# Main brain of AI, contains policy/target NN as well as training functions


class DDQN(object):
	def __init__(self, env, device):
		self.device = device
		self.state_shape = env.observation_space.shape
		self.num_actions = env.action_space.n
		self.policy_net = CnnDQN(self.state_shape, self.num_actions).to(device)
		self.target_net = CnnDQN(self.state_shape, self.num_actions).to(device)
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
		self.loss_fn = nn.MSELoss()
		self.epsilon = EPSILON_START
		self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)

		self.update_target()

	# a way to update the learning_rate, not used atm, might cause memory issues (find a better way)
	def update_learning_rate(learning_rate):
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

	# Each frame I choose an action with epsilon greedy strategy. I either
	# exploit (choose action with best q value), or explore (choose random action)
	def choose_action(self, state):
		if random.random() > self.epsilon:  # exploit
			state = torch.FloatTensor(np.float32(state)).to(device).unsqueeze(0)
			q_value = self.policy_net(state).detach()
			action = q_value.max(1)[1].item()
		else:  # explore
			action = random.randrange(self.num_actions)
		return action

	# every [TARGET_REPLACE_ITER] frames, target_net's weights are updated to the policy net
	def update_target(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())

	def train(self):
		state, action, reward, next_state, done = self.replay_buffer.sample()

		state = torch.FloatTensor(np.float32(state)).to(device)
		next_state = torch.FloatTensor(np.float32(next_state)).to(device)
		action = torch.LongTensor(action).to(device)
		reward = torch.FloatTensor(reward).to(device)
		done = torch.FloatTensor(done).to(device)

		# todo check if I should be detaching (detaching seems to decrease performance, find out why)
		q_values = self.policy_net(state)
		next_q_values = self.policy_net(next_state).detach()
		next_q_state_values = self.target_net(next_state).detach()

		# calculate q value
		q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
		# calculate target q_value
		next_q_value = next_q_state_values.gather(
			1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
		target_q_value = reward + GAMMA * next_q_value * (1 - done)

		# loss_fn used is in ddqn __init__
		loss = self.loss_fn(q_value, target_q_value)

		# backpropagate loss (check if it is being backpropogated to the target network as well)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss

# plots reward/loses per episode and stores it in an image

def plot(frame_idx, rewards, losses):
	plt.figure(figsize=(20, 5))
	plt.subplot(131)
	plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
	plt.plot(rewards)
	plt.subplot(132)
	plt.title('loss')
	# plot last 50000 losses
	plt.plot(losses[-1000000:])
	plt.savefig('plots/' + str(frame_idx) + '.jpg')
	# closes all plot data to release memory
	plt.cla()
	plt.clf()
	plt.close('all')

if __name__ == "__main__":
    # create environment
    env = make_atari(ENV_ID)
    # wrap environment
    env = wrap_deepmind(env, frame_stack=True)
    #env = wrap_pytorch(env)

    # .to(device) sends data to gpu for processing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ddqn = DDQN(env, device)

    # analytics data
    losses = []
    all_rewards = []
    episode_reward = 0

    start_time = time.time()
    state = env.reset()

    prev_action = 0
    for frame_idx in range(1, NUM_FRAMES + 1):
        # calculate current epsilon value and update ddqn's epsilon
        ddqn.epsilon = epsilon_by_frame(frame_idx)
        # choose action epsilon greedy
        action = ddqn.choose_action(state)

        # execute action, and store results in replay memory
        state_, reward, done, _ = env.step(action)
        if action != prev_action:
        	reward -= 0.1
        prev_action = action
        ddqn.replay_buffer.push(state, action, reward, state_, done)

        state = state_
        episode_reward += reward

        # episode is done
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        # if I have enough replay memory stored, start training
        if len(ddqn.replay_buffer) > REPLAY_INITIAL:
            if frame_idx % TRAINING_FREQUENCY == 0:
                loss = ddqn.train()
                losses.append(loss.item())

        # every 10000 frames, store analytics, along with model.state_dict()
        if frame_idx % ANALYTICS_FREQUENCY == 0:
            plot(frame_idx, all_rewards, losses)
            torch.save(ddqn.policy_net.state_dict(),
                    'models/' + 'policy_' + str(frame_idx))
            print(f'current training time: {time.time() - start_time} seconds')

        # update target_net weights with policy_net weights
        if frame_idx % TARGET_REPLACE_ITER == 0:
            ddqn.update_target()

        if frame_idx % 1000 == 0:
            print(frame_idx)

        # caused memory error
        #if frame_idx % REPLAY_BUFFER_BACKUP_FREQUENCY == 0:
        #	with open('replay_buffer.pkl', 'wb') as f:
        #		pickle.dump(ddqn.replay_buffer, f)

    print(f'training took {time.time() - start_time} seconds')
