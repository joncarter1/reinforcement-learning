import numpy as np
from SafeRL.ModelLearning import create_model, NNModel
from SafeRL.MPC import MPCLearner
import gym
import pickle
from copy import deepcopy
from collections import deque
import random
from SafeRL.ModelLearning import gaussian_nll
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"gaussian_nll": gaussian_nll})


def zero_func(x):
    return 0


class MPCLearner2:
    def __init__(self, state_dims=10, action_dims=2, probabilistic=True, reward_func=None, cost_func=None, final_cost=None, value_func=None):
        self.MEMORY_SIZE = 100000
        self.MIN_REPLAY_MEMORY_SIZE = 10000  # Minimum number of steps in a memory to start training
        self.REPLAY_MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.MINIBATCH_SIZE = 512
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.combined_dims = self.state_dims + self.action_dims
        self.policy_dims = self.state_dims
        self.neurons = 500
        self.layers = 2
        self.B = 5
        self.bootstrap_data_inds = [[] for _ in range(self.B)]  # Track samples used by each bootstrap NN
        dropout = 0
        l2_penalty = 0
        self.probabilistic = probabilistic
        self.dynamics_model = NNModel(input_size=self.combined_dims,
                                       output_size=self.state_dims,
                                       probabilistic=probabilistic,
                                       b=self.B, output_activation="linear",
                                       neurons=self.neurons, layers=self.layers,
                                       dr=dropout, l2_penalty=l2_penalty)
        self.input_mean, self.input_std, self.output_mean, self.output_std = [0, 1, 0, 1]
        self.normalised = False
        self.safety_threshold = 100
        self.reward_func = reward_func
        self.min_sigmas = [None for _ in range(self.B)]
        self.max_sigmas = [None for _ in range(self.B)]
        self.cost_func = cost_func if cost_func else zero_func
        self.final_cost = final_cost if final_cost else zero_func
        self.value_func = value_func if cost_func else create_model(input_size=self.state_dims, output_size=1,
                                                                    output_activation="linear", neurons=self.neurons,
                                                                    layers=self.layers, dr=dropout,
                                                                    l2_penalty=l2_penalty)

    def set_vars(self):
        """Get variance limits on training data, for clipping test data.
            Technique taken from Chua et al 2018: Deep Reinforcement Learning in a Handful of Trials."""
        state_action_vector = np.array(self.REPLAY_MEMORY)[:, :self.combined_dims]
        normalised_inputs = (state_action_vector - self.input_mean) / self.input_std
        for b_i in range(self.B):
            bootstrap_data = normalised_inputs[self.bootstrap_data_inds[b_i]]
            normalised_outputs = self.dynamics_model.predict(bootstrap_data, b_i)
            sigmas = np.exp(normalised_outputs[:, self.state_dims:])
            self.min_sigmas[b_i] = np.min(sigmas, axis=0)
            self.max_sigmas[b_i] = np.max(sigmas, axis=0)
        return

    def model_prediction(self, robot_state, action, b_i=0):
        state_action = np.expand_dims(np.hstack((robot_state, action)), axis=1).T
        normalised_input = (state_action - self.input_mean) / self.input_std
        normalised_output = self.dynamics_model.predict(normalised_input, b_i)
        if self.probabilistic:
            mean = normalised_output[:, :self.state_dims] * self.output_std + self.output_mean
            sigma = np.exp(normalised_output[:, self.state_dims:])
            return mean, sigma
        else:
            return normalised_output * self.output_std + self.output_mean

    def normalise_buffer(self):
        return
        print("Normalising buffer for training")
        buffer_array = np.array(self.REPLAY_MEMORY)
        self.input_mean = np.mean(buffer_array[:, :self.combined_dims], axis=0)
        self.input_std = np.std(buffer_array[:, :self.combined_dims], axis=0)
        self.output_mean = np.mean(buffer_array[:, -self.state_dims:], axis=0)
        self.output_std = np.std(buffer_array[:, -self.state_dims:], axis=0)
        stacked_mean = np.hstack((self.input_mean, self.output_mean))
        stacked_std = np.hstack((self.input_std, self.output_std))
        buffer_array = (buffer_array - stacked_mean) / stacked_std
        self.REPLAY_MEMORY = list(buffer_array)

    def store_transition(self, robot_state, action, new_robot_state):
        state_action = np.hstack((robot_state.reshape(1, self.state_dims), action.reshape(1, self.action_dims)))
        normalised_state_action = (state_action - self.input_mean) / self.input_std
        delta_state = (new_robot_state - robot_state).reshape(1, self.state_dims)
        normalised_delta_state = (delta_state - self.output_mean) / self.output_std
        stacked_transition = np.squeeze(np.hstack((normalised_state_action, normalised_delta_state)))
        self.REPLAY_MEMORY.append(stacked_transition)
        bootstrap_inds = np.random.binomial(1, 1, 5)
        new_index = len(self.REPLAY_MEMORY) - 1
        for b_i in range(self.B):
            if bootstrap_inds[b_i] == 1:
                self.bootstrap_data_inds[b_i].append(new_index)
        return

    def train_model(self, epochs=5):
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        if not self.normalised:
            self.normalise_buffer()
            self.normalised = True

        for b_i in range(self.dynamics_model.B):
            """Train each bootstrapped neural network"""
            data_sample = np.array(self.REPLAY_MEMORY)[self.bootstrap_data_inds[b_i]]
            state_action_vector = data_sample[:, :self.combined_dims]
            delta_state_vector = data_sample[:, -self.state_dims:]

            self.dynamics_model.train(x=state_action_vector, y=delta_state_vector, b_i=b_i, batch_size=self.MINIBATCH_SIZE,
                                      verbose=1, epochs=epochs, validation_split=0.1, shuffle=True)

        self.set_vars()  # Update variance limits after training
        return

    def save(self, model_name):
        pickle.dump(self, open(model_name, "wb"))


class NNPolicy:
    layers = 2
    neurons = 500
    dr = 0.0
    l2_penalty = 0.0
    lr = 0.001

    def __init__(self, state_dims, action_dims, existing_model=None):
        self.state_dims = state_dims
        self.action_dims = action_dims
        if not existing_model:
            self.model = create_model(input_size=self.state_dims,
                                       output_size=self.action_dims,
                                       output_activation="tanh",
                                       neurons=self.neurons,
                                       layers=self.layers,
                                       dr=self.dr,
                                       l2_penalty=self.l2_penalty,
                                       lr=self.lr)
        else:
            self.model = existing_model

    def __call__(self, state):
        return self.model.predict(state)


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")

    nn_policy = NNPolicy(2, 1)
    load = False
    if load:
        mpc_learner = pickle.load(open("MountainCarPE", "rb"))
    else:
        mpc_learner = MPCLearner2(2, 1, False)

    render = False

    for i in range(11):
        state = env.reset()
        done = False
        if render:
            env.render()
        while not done:
            #action = nn_policy(state.T)
            action = np.random.uniform(-1, 1, size=(1,))
            new_state, reward, done, info = env.step(action)
            if done:
                print(state, new_state)
            if render:
                env.render()
            mpc_learner.store_transition(state, action, new_state)
            state = new_state
        print(f"Episode {i}")

    mpc_learner.train_model(10)
    print(f"min sigmas {mpc_learner.min_sigmas}")
    print(f"max sigmas {mpc_learner.max_sigmas}")

    mpc_learner.save("MountainCarPE")
    print(mpc_learner.input_std)
    print(mpc_learner.output_std)
    print("done")


