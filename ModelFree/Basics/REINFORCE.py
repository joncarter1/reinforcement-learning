import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Sequential, Model, load_model, model_from_json
from keras.optimizers import Adam
import keras.backend as K
from collections import deque

from tqdm import tqdm
env = gym.make('LunarLander-v2')
#env = gym.make('Acrobot-v1')


def advantage_loss(advantages):
    def custom_loss(y_true, y_pred):
        out = K.clip(y_pred, 1e-8, 1 - 1e-8)  # Numerical stability of log
        ll = y_true * K.log(out)
        return K.sum(-ll * advantages)
    return custom_loss

class PolicyEstimator:
    def __init__(self, env, name=None):
        # Environment specifics
        self.env = env
        self.gamma = 0.97
        self.G = 0
        self.input_shape = env.observation_space.shape
        self.n_outputs = env.action_space.n
        self.action_space = [i for i in range(self.n_outputs)]
        # Neural network architecture
        self.l1_dims = 50
        self.l2_dims = 50
        self.lr = 0.001
        # Memory
        self.states, self.actions, self.rewards = [], [], []
        # Stats for tracking performance
        self.scores = []
        self.weights1 = []  # First weight matrix
        self.weights2 = []  # Second weight matrix

        self.policy_nn, self.predict_nn = self.create_model()

        if name:
            self.policy_nn.load_weights(f"{name}/PolicyNN-weights.h5")
            self.predict_nn.load_weights(f"{name}/PredictNN-weights.h5")

    def create_model(self):
        input = Input(shape=self.input_shape)
        advantages = Input(shape=[1])
        dense1 = Dense(self.l1_dims, input_shape=(2,),activation="relu")(input)
        dense2 = Dense(self.l2_dims, input_shape=(2,), activation="relu")(dense1)
        probs = Dense(self.n_outputs, activation="softmax")(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)  # Numerical stability of log
            ll = y_true*K.log(out)
            return K.sum(-ll*advantages)

        policy = Model(input=[input, advantages], output=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)

        predict = Model(input=[input], output=[probs])

        return policy, predict

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.predict_nn.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_transitions(self):
        self.states, self.actions, self.rewards = [], [], []

    def learn(self):
        state_memory = np.array(self.states)
        action_memory = np.array(self.actions)
        reward_memory = np.array(self.rewards)
        # One hot encoding of actions
        actions = np.zeros([len(action_memory), self.n_outputs])
        actions[np.arange(len(action_memory)), action_memory] = 1
        G = np.zeros_like(reward_memory)  #Future reward vector
        for t in range(len(reward_memory)):
            G_t = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_t += reward_memory[k]*discount
                discount *= self.gamma
            G[t] = G_t

        # Baselines
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean)/std
        cost = self.policy_nn.train_on_batch([state_memory, self.G], actions)
        self.clear_transitions()
        return cost

    def update_stats(self, score):
        self.scores.append(score)
        self.weights1.append(self.policy_nn.get_weights()[2])
        self.weights2.append(self.policy_nn.get_weights()[4])

    def save_model(self, name):
        self.policy_nn.save_weights(f"{name}/PolicyNN-weights.h5")
        self.predict_nn.save_weights(f"{name}/PredictNN-weights.h5")
        return 'Saved model weights'

    def load_model(self, name):
        policy_nn, predict_nn = self.create_model()
        policy_nn.set_weights()
        return 'Loaded models'

    def save_all(self, name):
        if not os.path.isdir(name):
            os.makedirs(name)
        pickle.dump(self.weights1, open(f"{name}/weights1.p", "wb"))
        pickle.dump(self.weights2, open(f"{name}/weights2.p", "wb"))
        pickle.dump(self.scores, open(f"{name}/scores.p", "wb"))
        self.save_model(name)


if __name__ == "__main__":
    #from gym.envs.box2d.lunar_lander import demo_heuristic_lander, LunarLander
    #demo_heuristic_lander(LunarLander(), render=True)

    LOAD_MODEL = None
    LOAD_MODEL = "Discount"

    if LOAD_MODEL:
        agent = PolicyEstimator(env, LOAD_MODEL)
    else:
        agent = PolicyEstimator(env)

    EPISODES = 3000
    # Iterate over episodes

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        show = False
        if not episode % 1000 or LOAD_MODEL:
            show = True
        current_state = env.reset()
        score = 0
        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            action = agent.choose_action(current_state)
            next_state, reward, done, _ = env.step(action)
            if show:
                env.render()
            agent.store_transition(current_state, action, reward)
            current_state = next_state
            score += reward
        agent.update_stats(score)

        agent.learn()

    #agent.save_all('Discount')
