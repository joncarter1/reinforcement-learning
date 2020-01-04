import numpy as np

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 20:00:08 2019

@author: Jonathan
"""

import gym
from gym.wrappers import Monitor
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
step_size = 40
l = 0.1
discount = 0.97
episodes = 3000
show_every = 1000
window = 100

e = 0.2
e_decay = 0.999

discrete_statespace = [step_size] * len(env.observation_space.high)
discrete_step = (env.observation_space.high - env.observation_space.low) / discrete_statespace
LOAD_TABLE = False

if LOAD_TABLE:
    q_table = np.load('q_table.npy')
    ucb_table = np.load('ucb_table.npy')
else:
    q_table = np.random.uniform(low=-2, high=0, size=(discrete_statespace + [env.action_space.n]))
    count_table = np.ones(shape=(discrete_statespace + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [], 'e': []}


def discretize(state):
    discretized_state = (state - env.observation_space.low) / discrete_step
    return tuple(discretized_state.astype(np.int))

def ucb(t, x):
    return ((2*np.log(t))/x)**0.5

t = 1
for episode in range(episodes):
    episode_reward = 0
    render = False
    done = False
    discrete_state = discretize(env.reset())

    if not episode % show_every:
        render = True
    while not done:
        # Epsilon greedy policy
        action = np.argmax(q_table[discrete_state]+ucb(t, count_table[discrete_state]))

        new_state, reward, done, _ = env.step(action)
        t += 1
        episode_reward += reward
        new_discrete_state = discretize(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            count_table[discrete_state + (action,)] += 1
            new_q = (1 - l) * current_q + l * (reward + discount * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    e = max(e * e_decay, 0.05)
    ep_rewards.append(episode_reward)

    if not episode % window:
        average_reward = sum(ep_rewards[-window:]) / len(ep_rewards[-window:])
        min_reward = min(ep_rewards[-window:])
        max_reward = max(ep_rewards[-window:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['min'].append(min_reward)
        aggr_ep_rewards['max'].append(max_reward)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['e'].append(e)
        print(f"Episode: {episode}, max: {max_reward}, min :{min_reward}, avg: {average_reward}, e: {e}")

fig, ax1 = plt.subplots()

ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward')
ax1.tick_params(axis='y')

ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max', color='yellow')
# ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min', color='green')
ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg', color='red')
ax1.legend()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['e'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
env.close()