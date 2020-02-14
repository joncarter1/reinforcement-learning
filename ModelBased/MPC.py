import warnings
warnings.filterwarnings('ignore')
import multiprocessing
import os
import pickle
import numpy as np
from autograd import grad
import autograd.numpy as np
from autograd.numpy import array as arr
import gym
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Sequential, Model, load_model, model_from_json
from keras.optimizers import Adam
import keras.backend as K
from collections import deque
from timeit import default_timer as timer
from tqdm import tqdm
from gym.envs.box2d.lunar_lander import LunarLander
import random
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from ModelFree.ValueMethods.QlearningMCar import QAgent
from policies import SoftmaxPolicy, GreedyPolicy, UCBPolicy

class PolicyEstimator:
    def __init__(self, env, name=None, gamma=0.99, baseline=False, normalise=True):
        # Environment specifics
        self.env = env
        self.normalise = normalise
        self.env_name = env.unwrapped.spec.id
        self.gamma = gamma
        self.baseline = baseline
        self.G_n, self.advantages = 0, 0
        self.state_space_shape = env.observation_space.shape
        self.n_outputs = env.action_space.n
        self.action_space = [i for i in range(self.n_outputs)]
        self.REPLAY_MEMORY_SIZE = 100000  # How many last steps to keep for model training]
        self.MIN_MEMORY_SIZE = 1000
        self.MINIBATCH_SIZE = 64
        # Neural network architecture
        self.dropout=False
        self.l1_dims = 150
        self.l2_dims = 150
        self.lr = 1e-3
        # Stats for tracking performance
        self.scores = []
        self.weights1 = []  # First weight matrix
        self.weights2 = []  # Second weight matrix
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)  # List / buffer
        self.model_nn, self.value_nn = self.create_model()


    def create_model(self):
        model_nn = Sequential()
        model_nn.add(Dense(self.l1_dims, input_shape=(3,)))
        model_nn.add(Activation("relu"))
        if self.dropout:
            model_nn.add(Dropout(0.1))
        model_nn.add(Dense(self.l2_dims))
        model_nn.add(Activation("relu"))
        if self.dropout:
            model_nn.add(Dropout(0.1))
        model_nn.add(Dense(2, activation="linear"))
        model_nn.compile(loss="mse", optimizer='rmsprop', metrics=["accuracy"])

        value_nn = Sequential()
        value_nn.add(Dense(self.l1_dims, input_shape=(3,)))
        value_nn.add(Activation("relu"))
        # model.add(Dropout(0.1))
        value_nn.add(Dense(self.l2_dims))
        value_nn.add(Activation("relu"))
        # model.add(Dropout(0.1))
        value_nn.add(Dense(2, activation="linear"))
        value_nn.compile(loss="mse", optimizer='adam', metrics=["accuracy"])

        return model_nn, value_nn

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.append([state, action, reward, next_state-state, done])

    def clear_transitions(self):
        self.states, self.actions, self.rewards = [], [], []

    def learn(self):
        if len(self.replay_memory) < self.MIN_MEMORY_SIZE:
            return
        minibatch = np.array(random.sample(self.replay_memory, self.MINIBATCH_SIZE))
        delta_states = np.array([transition[3] for transition in minibatch])
        model_inputs = np.array([np.hstack((transition[0], transition[1])) for transition in minibatch])

        self.model_nn.fit(model_inputs, delta_states, batch_size=self.MINIBATCH_SIZE,
                       verbose=False, shuffle=True)
        return

    def update_stats(self, score):
        self.scores.append(score)
        #self.weights1.append(self.policy_nn.get_weights()[2])
        #self.weights2.append(self.policy_nn.get_weights()[4])

    def save_model(self, name):
        self.policy_nn.save_weights(f"{name}/PolicyNN-weights.h5")
        self.predict_nn.save_weights(f"{name}/PredictNN-weights.h5")
        if self.value_nn:
            self.value_nn.save_weights(f"{name}/ValueNN-weights.h5")
        return 'Saved model weights'

    def save_all(self, name):
        if not os.path.isdir(self.env_name):
            os.makedirs(self.env_name)
        subdir_name = "/".join([self.env_name, name])
        if not os.path.isdir(subdir_name):
            os.makedirs(subdir_name)
        #pickle.dump(self.weights1, open(f"{subdir_name}/weights1.p", "wb"))
        #pickle.dump(self.weights2, open(f"{subdir_name}/weights2.p", "wb"))
        pickle.dump(self.scores, open(f"{subdir_name}/scores.p", "wb"))
        with open(f"{subdir_name}/specs.txt", "a") as text_file:
            text_file.write(f"lr = {self.lr}\n")
            text_file.write(f"gamma = {self.gamma}\n")
            text_file.write(f"baseline = {self.baseline}\n")
            text_file.write(f"normalise = {self.normalise}\n")
        self.save_model(subdir_name)


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = PolicyEstimator(env)
    policy = GreedyPolicy(0.2, 0.996)
    seed = 0
    q_agent = QAgent(env, policy, 0.1, 0.97, seed=seed, LOAD_TABLE=True)
    # Iterate over episodes
    EPISODES = 100
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
            action = env.action_space.sample()
            #action = q_agent.choose_action(current_state, episode)
            new_state, reward, done, _ = env.step(action)

            q_agent.update_table(episode, done, current_state, action, reward, new_state)
            episode_reward += reward

            agent.store_transition(current_state, action, reward, new_state, done)
            agent.learn()
            current_state = new_state
        print(episode_reward)
        print(current_state)
    print(len(agent.replay_memory))

    print("ep")
    current_states = np.array([transition[0] for transition in agent.replay_memory])
    delta_states = np.array([transition[3] for transition in agent.replay_memory])
    actions = np.array([transition[1] for transition in agent.replay_memory])
    predicted_states = []

    i = 0
    while not agent.replay_memory[i][4]:
        i += 1
    i += 1
    i_start = dc(i)
    predictive_state = current_states[i].reshape(1, 2)
    predicted_states.append(dc(predictive_state))
    while not agent.replay_memory[i][4]:
        action = actions[i].reshape(-1,1)
        model_input = np.hstack((predictive_state, action))
        predictive_state = agent.model_nn.predict(model_input)+predictive_state
        predicted_states.append(dc(predictive_state))
        i += 1
    i_end = dc(i)
    predicted_states = np.array(predicted_states).squeeze()
    lim = 200

    plt.figure()
    plt.plot(current_states[i_start:i_end,0], label="True")
    plt.plot(predicted_states[:, 0], label="Propagated")
    plt.title("Positions")
    plt.ylim(-1.2,0.6)
    plt.legend()

    plt.figure(2)
    plt.title("Velocities")
    plt.plot(current_states[i_start:i_end, 1], label="True")
    plt.plot(predicted_states[:, 1], label="Propagated")
    plt.ylim(-0.1, 0.1)
    plt.legend()

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    predictive_state = dc(current_state).reshape(1, 2)
    predicted_states2 = []
    predicted_states2.append(predictive_state)
    new_states = [dc(current_state)]
    # Reset flag and start iterating until episode ends
    done = False

    while not done:
        # Epsilon greedy exploration policy
        # action = env.action_space.sample()
        action = q_agent.choose_action(current_state, episode)
        new_state, reward, done, _ = env.step(action)
        model_input = np.hstack((predictive_state, np.array(action).reshape(-1, 1)))
        predictive_state = agent.model_nn.predict(model_input) + predictive_state
        predicted_states2.append(dc(predictive_state))
        q_agent.update_table(episode, done, current_state, action, reward, new_state)
        episode_reward += reward

        agent.store_transition(current_state, action, reward, new_state, done)
        agent.learn()
        current_state = new_state
        new_states.append(dc(current_state))
    print(episode_reward)
    print(current_state)

    predicted_states2 = np.squeeze(np.array(predicted_states2))
    new_states = np.squeeze(np.array(new_states))
    plt.figure()
    plt.plot(new_states[:, 0], label="True")
    plt.plot(predicted_states2[:, 0], label="Propagated")
    plt.title("Positions")
    plt.ylim(-1.2, 0.6)
    plt.legend()

    plt.figure()
    plt.title("Velocities")
    plt.plot(new_states[:, 1], label="True")
    plt.plot(predicted_states2[:, 1], label="Propagated")
    plt.ylim(-0.1, 0.1)
    plt.legend()
    plt.show()
