
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
import random, math, time

# hyperparameters
gamma = 0.99  # discount factor for reward
resume = True  # resume from previous checkpoint?
render = False
MAX_FRAMES = 5000000
eps = np.finfo(np.float32).eps.item()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CnnPGN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnPGN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []


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


def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    # plot last 50000 losses
    plt.plot(losses[-50000:])
    plt.savefig('plots/' + str(frame_idx) + '.jpg')
    # closes all plot data to release memory
    plt.cla()
    plt.clf()
    plt.close('all')

def select_action(policy, state, device):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state = state.view((1,1,80,80)).to(device)
    probs = policy(state)
    #print("Probs = ", probs)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    #print(action)
    return action.item()


def finish_episode(policy, optimizer, device, losses):

    R = 0
    policy_loss = []
    returns = []
    #print(policy.rewards)
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    #print(returns)
    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std())
    for log_prob, R in zip(policy.saved_log_probs, returns):
        #print('{} * {} = {}'.format(-log_prob, R, -log_prob * R))
        policy_loss.append(-log_prob * R)
    #print()
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    losses.append(policy_loss)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def play_game():
    render = False
    env = gym.make("Pong-v0")

    action_space_size = 4
    losses = []
    all_rewards = []
    current_frame = env.reset()
    current_frame = prepro(current_frame)

    #print(current_frame.shape)

    previous_frame = None
    running_reward = 10
    reward_sum = 0
    episode_num = 0
    frame_count = 1
    wins, batch_games_played = 0.0, 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = CnnPGN(current_frame.shape, action_space_size).to(device)
    optimizer = optim.RMSprop(policy.parameters(), lr=2e-4)
    policy.load_state_dict(torch.load('models\policy_1560000')) # Currently about 1000 episodes played through
    print("Starting Up:")
    start_time = time.time()

    while episode_num < 50000:
        if render:
            env.render()



        # if(frame_count % 1000 == 0):
        #     print('Time to train on {} frames: {} seconds'.format(frame_count, time.time() - start_time))
        #     start_time = time.time()

        difference_image = current_frame - \
            previous_frame if previous_frame is not None else np.zeros_like(current_frame)
        previous_frame = current_frame
        #print(difference_image.shape, np.amax(difference_image))

        action = select_action(policy, difference_image, device)
        current_frame, reward, done, _ = env.step(action)
        #print("Reward = ", reward)
        reward_sum += reward

        if(reward != 0.0):
            batch_games_played += 1
            if (reward == 1.0):
                wins += 1.0

        current_frame = prepro(current_frame)

        policy.rewards.append(reward)

        if (frame_count % 10000 == 0):
            plot(frame_count, all_rewards, losses)
            print("Time passed = {} seconds." \
                  " Episode = {}. Most recent avg loss is {}. Wins {} / {} since last update. Ratio = {}".format(time.time()\
                - start_time, episode_num, sum(losses[-5:]) / 5, wins, batch_games_played, wins/batch_games_played))
            wins = 0
            batch_games_played = 0
            #render = False

        if (frame_count % 50000 == 0):
            print("Saving model.")
            torch.save(policy.state_dict(),
                       'models/' + 'policy_' + str(frame_count))
            #render = True

        if done:

            #print("Num wins:", wins)
            episode_num += 1
            all_rewards.append(reward_sum)

            running_reward = 0.05 * reward_sum + (1 - 0.05) * running_reward
            finish_episode(policy, optimizer, device, losses)

            # if episode_num % batch_size == 0:
            #
            #     print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            #     episode_num, reward_sum, running_reward))

            if frame_count >= MAX_FRAMES:
                print("Solved! Running reward is now {}".format(running_reward))
                print('Total time to train {} frames: {} seconds'.format(frame_count, time.time() - start_time))
                print('{} episodes trained.'.format(episode_num))
                break



            previous_frame = None
            running_reward = 10
            reward_sum = 0

            #current_frame = env.reset()
            current_frame = prepro(env.reset())

        frame_count += 1

    print("Finished!!!")
play_game()


