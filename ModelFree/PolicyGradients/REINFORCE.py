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
from gym.envs.box2d.lunar_lander import LunarLander

class FFLunarLander(LunarLander):
    # Finite fuel Lunar Lander
    def __init__(self):
        LunarLander.__init__(self)
        self.fuel = 100

    def reset(self):
        super().reset()
        self.fuel = 100

    def step2(self, action):
        self.fuel = max(0, self.fuel - 1)
        if self.fuel > 0:
            return super().step(action)
        else:
            return super().step(0)



class PolicyEstimator:
    def __init__(self, env, name=None, gamma=0.99, baseline=False, normalise=True):
        # Environment specifics
        self.env = env
        self.normalise = normalise
        self.env_name = env.unwrapped.spec.id
        self.gamma = gamma
        self.baseline = baseline
        self.G_n, self.advantages = 0, 0
        self.input_shape = env.observation_space.shape
        self.n_outputs = env.action_space.n
        self.action_space = [i for i in range(self.n_outputs)]
        # Neural network architecture
        self.l1_dims = 50
        self.l2_dims = 50
        self.lr = 1e-3
        # Memory
        self.states, self.actions, self.rewards = [], [], []
        # Stats for tracking performance
        self.scores = []
        self.weights1 = []  # First weight matrix
        self.weights2 = []  # Second weight matrix

        if baseline:
            self.policy_nn, self.predict_nn, self.value_nn = self.create_model(True)
        else:
            self.policy_nn, self.predict_nn = self.create_model()
            self.value_nn = None

        if name:
            subdir_name = "/".join([self.env_name, name])
            self.scores = pickle.load(open(f"{subdir_name}/scores.p","rb"))
            self.policy_nn.load_weights(f"{subdir_name}/PolicyNN-weights.h5")
            self.predict_nn.load_weights(f"{subdir_name}/PredictNN-weights.h5")
            if baseline:
                self.value_nn.load_weights(f"{subdir_name}/ValueNN-weights.h5")

    def create_model(self, baseline=False):
        input = Input(shape=self.input_shape)
        advantages = Input(shape=[1])
        dense1 = Dense(self.l1_dims, activation="relu")(input)
        dense2 = Dense(self.l2_dims, activation="relu")(dense1)
        probs = Dense(self.n_outputs, activation="softmax")(dense2)

        if baseline:
            v_input = Input(shape=self.input_shape)
            v_dense1 = Dense(self.l1_dims, activation="relu")(input)
            v_dense2 = Dense(self.l2_dims, activation="relu")(v_dense1)
            v_out = Dense(1, activation="linear")(v_dense2)
            value_nn = Model(input=[input], output=[v_out])
            value_nn.compile(optimizer=Adam(lr=self.lr), loss='mse')

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)  # Numerical stability of log
            ll = y_true*K.log(out)
            return K.sum(-ll*advantages)

        policy = Model(input=[input, advantages], output=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)

        predict = Model(input=[input], output=[probs])

        if baseline:
            return policy, predict, value_nn
        else:
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
        discount_filter = np.flip(self.gamma**np.arange(0, reward_memory.shape[0]))
        G = np.convolve(reward_memory, discount_filter)[-reward_memory.shape[0]:]

        if self.normalise:
            mean = np.mean(G)
            std = np.std(G) if np.std(G) > 0 else 1
            self.G_n = (G - mean) / std  # Train on normalized values
        else:
            self.G_n = G

        if self.baseline:
            self.advantages = self.G_n - np.squeeze(self.value_nn.predict(state_memory))
            self.value_nn.train_on_batch(state_memory, self.G_n)
        else:
            self.advantages = self.G_n

        cost = self.policy_nn.train_on_batch([state_memory, self.advantages], actions)
        self.clear_transitions()
        return cost

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


def main(EPISODES=3000, LOAD_MODEL=None, save_name=None, baseline=True, normalise=False, gamma=0.99, seed=1):
    env = gym.make('LunarLander-v2')
    # env = LunarLander()
    # acrobot_env = gym.make('Acrobot-v1')
    np.random.seed(seed)
    env.seed(seed)
    # LOAD_MODEL = "NB0.99"
    gamma = 0.99
    use_b = baseline
    agent = PolicyEstimator(env, LOAD_MODEL, gamma, use_b, normalise)

    # Iterate over episodes

    steps = []
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
            # fuel = fuel+1 if action != 0 else fuel
            # _action = action if fuel < limit else 0  # One line hack for introducing fuel
            next_state, reward, done, _ = env.step(action)
            if show:
                env.render()
            agent.store_transition(current_state, action, reward)
            current_state = next_state
            score += reward
        agent.update_stats(score)
        ep_time = timer() - start
        steps.append(no_steps)
        start = timer()
        agent.learn()
        training_time = timer() - start

        if not episode % 200:
            print(ep_time, "seconds of episode")
            print(training_time, "to train")
            print(f"{score} score, {np.mean(steps[-100:])} mean no. steps")
    if save_name:
        agent.save_all(save_name)


if __name__ == "__main__":
    baseline, normalise, gamma = True, False, 0.95
    main(50, "BNM0.99", None, baseline=baseline, normalise=normalise, gamma=gamma)
