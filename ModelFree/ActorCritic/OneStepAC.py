import warnings
warnings.filterwarnings('ignore')
import multiprocessing
import os
import pickle
import numpy as np
import gym
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Sequential, Model, load_model, model_from_json
from keras.optimizers import Adam
import keras.backend as K
from collections import deque
from timeit import default_timer as timer
from tqdm import tqdm
from copy import deepcopy
import random

class PolicyEstimator:
    def __init__(self, env, name=None, gamma=0.99, normalise=True):
        # Environment specifics
        self.env = env
        self.normalise = normalise
        self.env_name = env.unwrapped.spec.id
        self.gamma = gamma
        self.G_n, self.advantages = 0, 0
        self.input_shape = env.observation_space.shape
        self.n_outputs = env.action_space.n
        self.action_space = [i for i in range(self.n_outputs)]
        # Neural network architecture
        self.l1_dims = 250
        self.l2_dims = 250
        self.lr = 1e-3
        # Memory
        self.states, self.actions, self.rewards, self.next_states = [], [], [], []
        self.REPLAY_MEMORY = deque(maxlen=10000)
        self.MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
        self.UPDATE_TARGET_EVERY = 5  # How often to sync. target and true value networks
        self.MINIBATCH_SIZE=32
        self.target_update_counter = 0
        # Stats for tracking performance
        self.scores = []
        self.weights1 = []  # First weight matrix
        self.weights2 = []  # Second weight matrix

        self.policy_nn, self.predict_nn, self.value_nn = self.create_models()
        self.target_value_nn = deepcopy(self.value_nn)  # Target network used for stabilisation

        if name:
            subdir_name = "/".join([self.env_name, name])
            self.scores = pickle.load(open("{}/scores.p","rb".format(subdir_name)))
            self.policy_nn.load_weights("{}/PolicyNN-weights.h5".format(subdir_name))
            self.predict_nn.load_weights("{}/PredictNN-weights.h5".format(subdir_name))

    def create_models(self):
        input = Input(shape=self.input_shape)
        advantages = Input(shape=[1])
        dense1 = Dense(self.l1_dims, activation="relu")(input)
        dense2 = Dense(self.l2_dims, activation="relu")(dense1)
        probs = Dense(self.n_outputs, activation="softmax")(dense2)

        #v_input = Input(shape=self.input_shape)
        v_dense1 = Dense(self.l1_dims, activation="relu")(input)
        v_dense2 = Dense(self.l2_dims, activation="relu")(v_dense1)
        v_out = Dense(1, activation="linear")(v_dense2)
        value_nn = Model(input=[input], output=[v_out])
        value_nn.compile(optimizer="rmsprop", loss='mse')

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)  # Numerical stability of log
            ll = y_true*K.log(out)
            return K.sum(-ll*advantages)

        policy = Model(input=[input, advantages], output=[probs])
        policy.compile(optimizer="rmsprop", loss=custom_loss) # Adam(lr=self.lr)

        predict = Model(input=[input], output=[probs])

        return policy, predict, value_nn

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.predict_nn.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def update_replay_memory(self, transition):
        self.REPLAY_MEMORY.append(transition)

    def store_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.update_replay_memory((state, action, reward, next_state, done))

    def clear_transitions(self):
        self.states, self.actions, self.rewards, self.next_states = [], [], [], []

    def learn(self):
        state_memory = np.array(self.states)
        action_memory = np.array(self.actions)
        reward_memory = np.array(self.rewards)
        next_state_memory = np.array(self.next_states)
        # One hot encoding of actions
        actions = np.zeros([len(action_memory), self.n_outputs])
        actions[np.arange(len(action_memory)), action_memory] = 1
        end = np.ones_like(reward_memory)
        end[-1] = 0

        # One step actor-critic td errors as in Sutton-Barto, we times by 'done' vector [1,1,1,...,1,0] so v(s_end) = 0
        value_targets = reward_memory + self.gamma*np.squeeze(self.value_nn.predict(next_state_memory))*end
        td_errors = value_targets - np.squeeze(self.value_nn.predict(state_memory))

        #self.value_nn.train_on_batch(state_memory, value_targets)
        self.policy_nn.train_on_batch([state_memory, td_errors], actions)

        self.clear_transitions()
        return

    def update_stats(self, score):
        self.scores.append(score)

    def train(self, terminal_state):
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = np.array(random.sample(self.REPLAY_MEMORY, self.MINIBATCH_SIZE))
        expand = lambda x: np.stack(x, axis=0)
        current_states = expand(minibatch[:, 0])
        rewards = expand(minibatch[:, 2])
        new_states = expand(minibatch[:, 3])
        end_states = expand(minibatch[:, 4])  # Value should be zero for end state.

        value_targets = rewards + self.gamma*np.squeeze(self.target_value_nn.predict(new_states))*(1-end_states)

        self.value_nn.fit(current_states, value_targets, batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Determine if we want to update target_model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_value_nn.set_weights(self.value_nn.get_weights())  # Synchronise neural nets
            self.target_update_counter = 0

    def save_model(self, name):
        self.policy_nn.save_weights("{}/PolicyNN-weights.h5".format(name))
        self.predict_nn.save_weights("{}/PredictNN-weights.h5".format(name))
        if self.value_nn:
            self.value_nn.save_weights("{}/ValueNN-weights.h5".format(name))
        return 'Saved model weights'

    def save_all(self, name):
        if not os.path.isdir(self.env_name):
            os.makedirs(self.env_name)
        subdir_name = "/".join([self.env_name, name])
        if not os.path.isdir(subdir_name):
            os.makedirs(subdir_name)
        #pickle.dump(self.weights1, open(f"{subdir_name}/weights1.p", "wb"))
        #pickle.dump(self.weights2, open(f"{subdir_name}/weights2.p", "wb"))
        pickle.dump(self.scores, open("{}/scores.p".format(subdir_name), "wb"))
        with open("{}/specs.txt".format(subdir_name), "a") as text_file:
            text_file.write("lr = {}\n".format(self.lr))
            text_file.write("gamma = {}\n".format(self.gamma))
            text_file.write("normalise = {}\n".format(self.normalise))
        self.save_model(subdir_name)


def main(EPISODES=3000, LOAD_MODEL=None, save_name=None,  normalise=False, gamma=0.99, seed=1):
    env = gym.make('LunarLander-v2')
    np.random.seed(seed)
    env.seed(seed)
    # LOAD_MODEL = "NB0.99"
    agent = PolicyEstimator(env, LOAD_MODEL, gamma, normalise)

    # Iterate over episodes

    steps = []
    scores = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        show = False
        if not episode % (EPISODES//10) and LOAD_MODEL:
            show = True
        current_state = env.reset()
        score, no_steps = 0, 0
        # Reset flag and start iterating until episode ends
        done = False
        start = timer()
        while not done:
            no_steps += 1  # Max steps
            action = agent.choose_action(current_state)
            next_state, reward, done, _ = env.step(action)
            if show:
                env.render()
            agent.store_transition(current_state, action, reward, next_state, done)
            if random.random() < 0.4:
                agent.train(done)
            current_state = next_state
            score += reward
        agent.update_stats(score)
        ep_time = timer() - start
        steps.append(no_steps)
        scores.append(score)
        start = timer()
        agent.learn()
        training_time = timer() - start

        if not episode % 10:
            print(ep_time, "seconds of episode")
            print(training_time, "to train")
            print("{} score, mean score {}, {} mean no. steps".format(score, np.mean(scores[-100:]), np.mean(steps[-100:])))
    if save_name:
        agent.save_all(save_name)


if __name__ == "__main__":
    normalise, gamma = False, 0.97
    load_model, save_model = "AC99", None
    load_model, save_model = None, "AC97"
    main(1500, load_model, save_model, normalise=normalise, gamma=gamma)
