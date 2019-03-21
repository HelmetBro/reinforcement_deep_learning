
import time
import math
import random
from itertools import count
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from pdb import set_trace
from collections import deque
import gym
import numpy as np
import _pickle as pickle
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

# hyperparameters
resume = True  # resume from previous checkpoint?
render = False
MAX_FRAMES = 2000000
eps = np.finfo(np.float32).eps.item()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnPGN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnPGN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        #print("Feeding forward:")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I


def select_action(policy, state, device):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state = state.view((1, 1, 80, 80)).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()


def play_game():
    render = True
    env = gym.make("Pong-v0")

    observation = env.reset()

    previous_frame = None
    episode_num = 0
    frame_count = 1
    current_frame = prepro(observation)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = CnnPGN(current_frame.shape, 4).to(device)
    policy.load_state_dict(torch.load('policy_1550000_Final'))

    print("Preparing Game:")
    start_time = time.time()

    while episode_num < 2:
        if render:
            env.render()

        if time.time() - start_time >= 10:
            env.close()
            return

        time.sleep(.005)
        difference_image = current_frame - \
            previous_frame if previous_frame is not None else np.zeros_like(
                current_frame)
        previous_frame = current_frame

        action = select_action(policy, difference_image, device)
        current_frame, reward, done, _ = env.step(action)

        current_frame = prepro(current_frame)

        if done:
            episode_num += 1
            previous_frame = None
            current_frame = prepro(env.reset())

        frame_count += 1

    print("Finished!!!")


play_game()
