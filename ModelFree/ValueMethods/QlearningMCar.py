#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 20:00:08 2019

@author: Jonathan
"""

from policies import SoftmaxPolicy, GreedyPolicy, UCBPolicy
import gym
import os
from gym.wrappers import Monitor
import pickle
import numpy as np
import matplotlib.pyplot as plt



class QAgent:
    def __init__(self, policy_func, l=0.1, discount=0.95, seed=0, LOAD_TABLE=False):
        self.env = gym.make('MountainCar-v0')
        self.env.seed(seed)
        self.policy = policy_func
        self.l = l
        self.discount = discount
        self.step_size = 20
        self.discrete_statespace = [self.step_size] * len(self.env.observation_space.high)
        self.discrete_step = (self.env.observation_space.high - self.env.observation_space.low) / self.discrete_statespace

        if LOAD_TABLE:
            self.q_table = np.load('q_table.npy')
        else:
            self.q_table = np.random.uniform(low=-2, high=0, size=(self.discrete_statespace + [self.env.action_space.n]))
        self.N_table = np.zeros_like(self.q_table)  # Actions chosen count

    def discretize(self, state):
        discretized_state = (state - self.env.observation_space.low) / self.discrete_step
        return tuple(discretized_state.astype(np.int))

    def choose_action(self, state, episode):
        discrete_state = self.discretize(state)
        action = self.policy.choose_action(self.q_table[discrete_state], self.N_table[discrete_state], episode)
        self.N_table[discrete_state + (action,)] += 1
        return action

    def update_table(self, episode, done, state, action, reward, new_state):
        discrete_state = self.discretize(state)
        if not done:
            new_discrete_state = self.discretize(new_state)
            max_future_q = np.max(self.q_table[new_discrete_state])
            current_q = self.q_table[discrete_state + (action,)]
            new_q = (1 - self.l) * current_q + self.l * (reward + self.discount * max_future_q)
            self.q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= self.env.goal_position:
            #print(f"Made it on episode {episode}")
            self.q_table[discrete_state + (action,)] = 0


def main(episodes, l, discount, strategy, parameter, seed=0):
    np.random.seed(seed)

    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [], 'parameter': []}

    show_every = 1000
    window = 250

    e = 1
    tau = 1

    if strategy == 'greedy':
        e_decay = parameter
        policy = GreedyPolicy(e, e_decay)
    elif strategy == 'softmax':
        tau_increase = parameter
        policy = SoftmaxPolicy(tau, tau_increase)
    else:
        raise ValueError

    q_agent = QAgent(policy, l, discount, seed=seed)

    for episode in range(0, episodes + 1):
        episode_reward = 0
        render = False
        done = False
        state = q_agent.env.reset()

        if not episode % show_every:
            render = True
        while not done:
            # Epsilon greedy policy
            action = q_agent.choose_action(state, episode)
            new_state, reward, done, _ = q_agent.env.step(action)
            episode_reward += reward
            if render:
                q_agent.env.render()
            q_agent.update_table(episode, done, state, action, reward, new_state)
            state = new_state
        ep_rewards.append(episode_reward)

        if not episode % (episodes // 10):
            average_reward = sum(ep_rewards[-window:]) / len(ep_rewards[-window:])
            min_reward = min(ep_rewards[-window:])
            max_reward = max(ep_rewards[-window:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['min'].append(min_reward)
            aggr_ep_rewards['max'].append(max_reward)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['parameter'].append(q_agent.policy.parameter)
            print(f"Episode: {episode}, max: {max_reward}, min :{min_reward}, avg: {average_reward}")
    """
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')
    ax1.tick_params(axis='y')
    ax1.plot(ep_rewards)
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max', color='yellow')
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min', color='green')
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg', color='red')

    ax1.legend()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['parameter'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()"""
    q_agent.env.close()
    if not os.path.isdir("results"):
        os.makedirs("results")
    if not os.path.isdir(f"results/{strategy}"):
        os.makedirs(f"results/{strategy}")
    pickle.dump([ep_rewards, aggr_ep_rewards, q_agent.q_table], open(f"results/{strategy}/{l}-{discount}-{parameter}.p", "wb"))

    # np.save('q_table', q_table)

if __name__ == "__main__":
    strategies = ['greedy', 'softmax']
    l, discount = 0.2, 0.97
    strategy = 'softmax'
    s_params = [[0.99, 0.995, 0.999], [1.0001, 1.001, 1.01]]
    ls = [0.05, 0.1, 0.2]
    l, discount = 0.2, 0.97
    discounts = [0.95,0.97]
    main(4000, l, discount, strategy, 1.01, seed=0)
