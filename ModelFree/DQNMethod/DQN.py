#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:24:01 2019

@author: Jonathan
"""

import tensorflow as tf
import pickle
from gym.wrappers import Monitor
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import keras.backend as K
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')
from collections import deque
import time
import numpy as np
from numpy import array as arr
import random

from tqdm import tqdm
import os
import gym


class DQNAgent:
    def __init__(self, env, gamma=0.99, model_name=None):
        self.env = env
        self.env_name = env.unwrapped.spec.id
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.gamma = gamma
        self.scores = []
        self.REPLAY_MEMORY_SIZE = 20000  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
        if model_name:
            subdir_name = "/".join([self.env_name, model_name])
            self.scores = pickle.load(open(f"{subdir_name}/scores.p","rb"))
            self.model.load_weights(f"{subdir_name}/weights.h5")
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(50, input_shape=self.env.observation_space.shape))
        model.add(Activation("relu"))
        #model.add(Dropout(0.1))
        model.add(Dense(50))
        model.add(Activation("relu"))
        #model.add(Dropout(0.1))
        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer='rmsprop', metrics=["accuracy"])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = arr(random.sample(self.replay_memory, self.MINIBATCH_SIZE))
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []
        
        for index, (current_state, action ,reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma*max_future_q
            else:
                new_q = reward
                
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=self.MINIBATCH_SIZE,
                       verbose=0, shuffle=False)
        
        # Determine if we want to update target_model
        if terminal_state:
            self.target_update_counter += 1
            
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def update_stats(self, score):
        self.scores.append(score)

    def save_model(self, name):
        self.model.save_weights(f"{name}/weights.h5")
        return 'Saved model weights'

    def save_all(self, name):
        if not os.path.isdir(self.env_name):
            os.makedirs(self.env_name)
        subdir_name = "/".join([self.env_name, name])
        if not os.path.isdir(subdir_name):
            os.makedirs(subdir_name)
        #pickle.dump(self.weights, open(f"{subdir_name}/weights1.p", "wb"))
        #pickle.dump(self.weights2, open(f"{subdir_name}/weights2.p", "wb"))
        pickle.dump(self.scores, open(f"{subdir_name}/scores.p", "wb"))
        with open(f"{subdir_name}/specs.txt", "a") as text_file:
            text_file.write(f"gamma = {self.gamma}\n")
            text_file.write(f"minibatch = {self.MINIBATCH_SIZE}\n")
            text_file.write(f"min. replay size = {self.MIN_REPLAY_MEMORY_SIZE}\n")
            text_file.write(f"replay size = {self.REPLAY_MEMORY_SIZE}\n")
        self.save_model(subdir_name)


def main(EPISODES, gamma, save_name=None, model_name=None, preview=False):

    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    EPSILON_DECAY = 0.996
    MIN_EPSILON = 0
    epsilon = epsilon*EPSILON_DECAY**1000
    print(epsilon)
    env = gym.make('LunarLander-v2')
    # env = Monitor(env, './video3')

    # For stats
    steps = []

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    agent = DQNAgent(env, gamma, model_name)

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            # Epsilon greedy exploration policy
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = env.action_space.sample()

            new_state, reward, done, lives = env.step(action)

            episode_reward += reward

            if preview and not episode % (EPISODES//5):
                env.render()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        agent.update_stats(episode_reward)
        steps.append(step)
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        if not episode % (EPISODES//10):
            print(f"{episode_reward} score, {np.mean(steps[-100:])} no. steps")

    if save_name:
        agent.save_all(save_name)


if __name__ == "__main__":
    main(10, 0.97, None, 'G0.993', True)
