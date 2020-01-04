#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:24:01 2019

@author: Jonathan
"""

import tensorflow as tf
from Peripherals import Blob, BlobEnv, ModifiedTensorBoard
from gym.wrappers import Monitor
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Permute
import keras.backend as K
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')
from collections import deque
import time
import numpy as np
import random

from tqdm import tqdm
import os
import gym


DISCOUNT = 0.98
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
SAVE_MODEL_EVERY = 2000
MODEL_NAME = 'acrobot4'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 10000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.15

#  Stats settings
LOAD_MODEL=None
#LOAD_MODEL="models/acrobot2___-88.00max_-252.16avg_-500.00min__1574342102.model"
LOAD_MODEL="models/2x256___-99.00max_-221.07avg_-500.00                                 min__1574218071.model"

if LOAD_MODEL:
    AGGREGATE_STATS_EVERY = 1  # episodes
else:
    AGGREGATE_STATS_EVERY = 1  # episodes

SHOW_PREVIEW = True

env = gym.make('Acrobot-v1')
#env = Monitor(env, './video3')

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class DQNAgent:
    def __init__(self, model=None):
        self.model= self.create_model()
        
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{MODEL_NAME}-{ts}".format(MODEL_NAME=MODEL_NAME, ts = int(time.time())))
        
        self.target_update_counter = 0
        

    def create_model(self):
        if LOAD_MODEL is not None:
            print("Loading model")
            model = load_model(LOAD_MODEL)
            print("Model loaded")
        else:
            model = Sequential()
            model.add(Dense(32, input_shape=env.observation_space.shape))
            model.add(Activation("relu"))
            model.add(Dropout(0.1))
            model.add(Dense(64))
            model.add(Activation("relu"))
            model.add(Dropout(0.1))
            model.add(Dense(env.action_space.n, activation="linear"))
            model.compile(loss="mse", optimizer='rmsprop', metrics=["accuracy"])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states = np.array([transition[0] for transition in minibatch])
        
        current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array([transition[3] for transition in minibatch])
        
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []
        
        for index, (current_state, action ,reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT*max_future_q
            else:
                new_q = reward
                
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
            
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        # Determine if we want to update target_model
        if terminal_state:
            self.target_update_counter += 1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    #agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # Epsilon greedy exploration policy
        if np.random.random() > 1:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = env.action_space.sample()

        new_state, reward, done, lives = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        if not episode % SAVE_MODEL_EVERY:
            agent.model.save('models/{MODEL_NAME}__{mr:_>7.2f}max_{avr:_>7.2f}avg_{minr:_>7.2f}min__{ts}.model'.format(MODEL_NAME=MODEL_NAME, mr=max_reward, avr=average_reward, minr=min_reward, ts=int(time.time())))

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
