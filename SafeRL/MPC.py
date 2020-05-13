import safety_gym
import gym
from gym import wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import random
import numpy as np
import pickle
from safety_gym.envs.engine import Engine
from tqdm import tqdm
import tensorflow as tf
from keras.models import load_model
import os
from copy import deepcopy
from collections import deque
from ModelLearning import NNModel, create_model
from Controller import human_policy, safe_policy
import matplotlib.pyplot as plt

"""
-ve rotation = clockwise torque
i.e. theta defined in the polar sense.
"""


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class MPCLearner:
    def __init__(self, state_dims=10, action_dims=2):
        self.MEMORY_SIZE = 100000
        self.MIN_REPLAY_MEMORY_SIZE = 10000  # Minimum number of steps in a memory to start training
        self.REPLAY_MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.MINIBATCH_SIZE = 512
        self.state_dims = 10
        self.hazard_dims = state_dims
        self.action_dims = action_dims
        self.combined_dims = self.state_dims + self.action_dims
        self.policy_dims = self.state_dims + self.hazard_dims
        self.neurons = 500
        self.layers = 2
        dropout = 0
        l2_penalty = 0
        self.dynamics_model = create_model(input_size=self.combined_dims,
                                           output_size=self.state_dims,
                                           output_activation="linear",
                                           neurons=self.neurons,
                                           layers=self.layers,
                                           dr=dropout,
                                           l2_penalty=l2_penalty)
        self.input_mean, self.input_std, self.output_mean, self.output_std = [0, 1, 0, 1]
        self.normalised = False
        self.safety_threshold = 3

    def model_prediction(self, robot_state, action):
        state_action = np.expand_dims(np.hstack((robot_state, action)), axis=1).T
        normalised_input = (state_action - self.input_mean) / self.input_std
        normalised_output = self.dynamics_model.predict(normalised_input)
        return normalised_output * self.output_std + self.output_mean

    def compute_trajectories(self, initial_robot_state, goal_pos, hazard_vector, policy):
        trajectory_length = 40   # MPC Horizon length
        no_trajectories = 100

        trajectory_costs = np.zeros(no_trajectories)
        trajectory_values = np.zeros(no_trajectories)

        current_robot_state = deepcopy(initial_robot_state)
        old_robot_state = current_robot_state
        policy_actions = []
        policy_reward = 0
        policy_cost = compute_cost(current_robot_state, goal_pos, hazard_vector)
        gamma = 0.97
        rf = 2  # Reward factor
        discount = 1
        for i in range(trajectory_length):
            next_action = policy(np.squeeze(current_robot_state), goal_pos, hazard_vector)
            policy_actions.append(next_action)
            delta_state = np.squeeze(self.model_prediction(current_robot_state, next_action))
            r_1 = deepcopy(current_robot_state[2])
            current_robot_state = old_robot_state + delta_state
            policy_cost = max(policy_cost, compute_cost(current_robot_state, goal_pos, hazard_vector))
            policy_reward += discount*compute_value(current_robot_state, old_robot_state)
            discount *= gamma
            old_robot_state = current_robot_state

        policy_actions = np.array(policy_actions).reshape(1, trajectory_length, self.action_dims)

        sigmas = np.squeeze(np.clip(np.std(policy_actions, axis=1), 1e-2, None))

        deltas = np.sum(np.abs(policy_actions[:, 1:, :]-policy_actions[:, :-1, :]), axis=1)
        length_scales = np.clip(np.squeeze(trajectory_length/(deltas/sigmas)), 0.01, trajectory_length)
        action_sequences = np.repeat(policy_actions, no_trajectories, axis=0) #.reshape(trajectory_length, no_trajectories, self.action_dims)

        correlated_noise = add_correlated_noise(sigmas, 2*length_scales)
        mpc_action_sequences = np.clip(action_sequences + correlated_noise, -1, 1)
        """dim = 0

        plt.figure()
        plt.plot(policy_actions[0, :, dim], label="Policy")
        plt.plot(mpc_action_sequences[6, :, dim], label="Exploration")
        plt.legend()
        plt.ylim(-1,1)
        plt.show()
        raise ValueError"""
        current_states = np.repeat(initial_robot_state.reshape(1, self.state_dims), no_trajectories, axis=0)
        old_robot_states = current_states
        discount = 1
        for i in range(trajectory_length):
            state_actions = np.hstack((current_states, mpc_action_sequences[:, i]))
            normalised_inputs = (state_actions - self.input_mean) / self.input_std
            normalised_outputs = self.dynamics_model.predict(normalised_inputs)
            delta_states = normalised_outputs*self.output_std + self.output_mean
            current_states = old_robot_states + delta_states
            state_costs = np.apply_along_axis(compute_cost, 1, current_states, goal_pos, hazard_vector)
            state_values = compute_values(current_states, old_robot_states)
            trajectory_costs = np.max((trajectory_costs, state_costs), axis=0)  # Infinite norm penalty on states
            trajectory_values += discount*state_values
            discount *= gamma
            old_robot_states = current_states

        bad_trajectories = np.where(trajectory_costs > policy_cost)
        trajectory_values[bad_trajectories] = -np.inf
        best_trajectory = np.argmax(trajectory_values)
        #print(f"Policy cost: {policy_cost}")
        if policy_reward > trajectory_values[best_trajectory]:
            #print("Overridden")
            return action_sequences[0, 0], 0
        else:
            #print("Best: ", best_trajectory, trajectory_values[best_trajectory], trajectory_costs[best_trajectory])

            if random.random() > 1:
                plt.figure()
                fig, axes = plt.subplots(2, 1)
                for i in range(2):
                    axes[i].plot(policy_actions[0, :, i], label="OG")
                    axes[i].plot(mpc_action_sequences[best_trajectory, :, i], label="Best")
                plt.legend()
                plt.show()
                raise ValueError
            return mpc_action_sequences[best_trajectory, 0], 1

    def normalise_buffer(self):
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
        state_action = np.hstack((robot_state, action))
        normalised_state_action = (state_action - self.input_mean) / self.input_std
        delta_state = new_robot_state - robot_state
        normalised_delta_state = (delta_state - self.output_mean) / self.output_std
        stacked_transition = np.hstack((normalised_state_action, normalised_delta_state))
        self.REPLAY_MEMORY.append(stacked_transition)
        return

    def train_on_batch(self):
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        if not self.normalised:
            self.normalise_buffer()
            self.normalised = True

        minibatch = np.array(random.sample(self.REPLAY_MEMORY, self.MINIBATCH_SIZE))

        state_action_vector = minibatch[:, :self.combined_dims]
        delta_state_vector = minibatch[:, -self.state_dims:]

        self.dynamics_model.fit(x=state_action_vector,
                                y=delta_state_vector,
                                batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False)

        return

    def train_model(self, epochs=5):
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        if not self.normalised:
            self.normalise_buffer()
            self.normalised = True

        state_action_vector = np.array(self.REPLAY_MEMORY)[:, :self.combined_dims]
        delta_state_vector = np.array(self.REPLAY_MEMORY)[:, -self.state_dims:]

        for i in range(epochs):
            noisy_input = state_action_vector + np.random.normal(0, 0.001, size=state_action_vector.shape)
            noisy_output = delta_state_vector + np.random.normal(0, 0.001, size=delta_state_vector.shape)

            self.dynamics_model.fit(x=noisy_input,
                                    y=noisy_output,
                                    batch_size=self.MINIBATCH_SIZE,
                                    verbose=1,
                                    epochs=1,
                                    validation_split=0.1,
                                    shuffle=True)
        return

    def save(self, model_name):
        pickle.dump(self, open(model_name, "wb"))


def add_correlated_noise(sigmas=[0.1], lengthscales=[0.1], trajectory_length=40, no_trajectories=100, uniform=True):
    """Add exploration noise to action sequences of each dimension."""
    mean = trajectory_length * [0]
    array_inds = np.arange(0, trajectory_length).reshape(trajectory_length, 1)
    correlations = []
    for (sigma, lengthscale) in zip(sigmas, lengthscales):
        if uniform:
            correlated_noise = np.random.uniform(-sigma, sigma, size=(no_trajectories, trajectory_length))
        else:
            cov_matrix = (sigma ** 2) * np.exp(-((array_inds.T - array_inds) / lengthscale) ** 2)
            correlated_noise = np.random.multivariate_normal(mean, cov_matrix, size=no_trajectories)
        correlations.append(correlated_noise)

    correlations = np.moveaxis(np.array(correlations), 0, -1)
    #print(correlations.shape)
    return correlations


def compute_cost(robot_state, goal_pos, hazards):
    sigma = 0.3
    displacements = hazards - (robot_state[:2]+goal_pos)
    distances = np.einsum('ij,ji->i', displacements, displacements.T)
    return np.max(np.exp(-(distances / sigma)**2))


def compute_value(robot_state, old_robot_state):
    return np.linalg.norm(old_robot_state[:2]) - np.linalg.norm(robot_state[:2])


def compute_values(robot_states, old_robot_states):
    return np.array([compute_value(robot_state, old_robot_state)
                     for (robot_state, old_robot_state)
                     in zip(robot_states, old_robot_states)])


def form_state(state_dict, position):
    modified_state_dict = deepcopy(state_dict)
    for key in ["magnetometer", "hazards_lidar"]:
        del modified_state_dict[key]  # Useless state measurement.
    modified_state_dict["gyro"] = modified_state_dict["gyro"][2]  # Just keep k value
    for key in ["accelerometer", "velocimeter"]:
        modified_state_dict[key] = modified_state_dict[key][:2]  # Remove z axis value
    stacked_state = np.hstack(list(modified_state_dict.values()))
    return np.hstack((position, stacked_state))


def save_results(memory_buffer):
    ensure_dir("policy_data")
    pickle.dump(memory_buffer, open(f"policy_data/replay_memory", "wb"))
    return True


def main(EPISODES, mpc_learner=None, render=False, save_name=None, policy=human_policy):
    if mpc_learner is None:
        mpc_learner = MPCLearner()

    #action_sequence = []

    for i in tqdm(range(EPISODES)):
        no_accepted = 0
        stored, total, episode_reward, episode_cost = 0, 0, 0, 0
        env_state = env.reset()
        position = env.robot_pos[:2] - env.goal_pos[:2]
        robot_state = form_state(env_state, position)
        if render:
            env.render()
        done = False
        while not done:
            regular = False
            hazards = env.gremlins_obj_pos + env.hazards_pos
            hazard_vector = np.array(hazards)[:, :2]
            goal_pos = env.goal_pos[:2]
            if regular:
                if np.random.random() < 0:
                    action = np.random.uniform(-1, 1, size=(2,))
                else:
                    action = policy(robot_state, env.goal_pos[:2], hazard_vector).reshape(2,)
            else:
                action, accepted = mpc_learner.compute_trajectories(robot_state, goal_pos, hazard_vector, policy)
                no_accepted += accepted
            new_env_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_cost += info["cost"]
            new_position = env.robot_pos[:2] - env.goal_pos[:2]
            new_robot_state = form_state(new_env_state, new_position)
            #print(f"State cost : {compute_cost(robot_state, goal_pos, hazard_vector  )}")
            if save_name:
                stored += 1
                mpc_learner.store_transition(robot_state, action, new_robot_state)
            total += 1
            if save_name and np.random.random() < 0.2:
                mpc_learner.train_on_batch()
            env_state = new_env_state
            robot_state = new_robot_state
            if render:
                env.render()

        print(f"Episode {i}, reward {episode_reward}, cost {episode_cost}, stored {stored}/{total}, accepted {no_accepted}/{total}")
        print(f"Buffer length: {len(mpc_learner.REPLAY_MEMORY)}")
        if save_name and i != 0 and not i % (EPISODES // 5):
            save_results(mpc_learner.REPLAY_MEMORY)
            mpc_learner.save("mpcmodel")

    #pickle.dump(action_sequence, open(f"action_sequence2", "wb"))
    if save_name:
        print(len(mpc_learner.REPLAY_MEMORY))
        save_results(mpc_learner.REPLAY_MEMORY)
        mpc_learner.save(save_name)
    return


if __name__ == "__main__":
    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'continue_goal': False,
        'observation_flatten': False,
        'observe_goal_dist': True,
        'observe_goal_comp': True,
        'observe_goal_lidar': False,
        'observe_hazards': True,
        'observe_vases': False,
        'observe_circle': True,
        'observe_vision': False,
        'observe_gremlins': False,
        'constrain_hazards': True,
        'constrain_vases': False,
        'constrain_gremlins': False,
        'lidar_max_dist': None,
        'lidar_num_bins': 32,
        'hazards_num': 10,
        'vases_num': 0,
        'gremlins_num': 0,
        'gremlins_travel': 0.5,
        'gremlins_keepout': 0.4,
    }
    env = Engine(config)

    training = False
    if training:
        ll = pickle.load(open("mpcmodel2", "rb"))
        model = create_model(input_size=ll.combined_dims,
                     output_size=ll.state_dims,
                     output_activation="linear",
                     dr=0.1, layers=2,
                     neurons=2000, lr=0.001,
                     l2_penalty=0)
        ll.dynamics_model = model
        ll.MINIBATCH_SIZE = 2048
        ll.train_model(epochs=25)
        # ll.save("mpcmodel3")
        #ll.train_model(epochs=20)
        ll.save("mpcmodel2")
        #main(EPISODES=50, mpc_learner=ll, render=False, policy=human_policy, save=True)
    else:
        loaded_learner = pickle.load(open("mpcmodel", "rb"))
        loaded_learner.safety_threshold = 0.8
        seeds = [18, 19, 20, 21, 22]
        for seed in seeds:
            env.seed(seed)  # 18 good
            np.random.seed(100)
            main(EPISODES=1, mpc_learner=loaded_learner, render=False, policy=human_policy, save_name=None)
