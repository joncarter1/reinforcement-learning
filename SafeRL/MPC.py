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
from Controller import human_policy, safe_policy, config
import matplotlib.pyplot as plt
"""
-ve rotation = clockwise torque
i.e. theta defined in the polar sense.
"""


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_env_state(robot_position, env_state_dict):
    """Pre process the environment state dictionary and robot position."""
    modified_state_dict = deepcopy(env_state_dict)
    for key in ["magnetometer"]:
        del modified_state_dict[key]  # Useless state measurement.
    modified_state_dict["gyro"] = modified_state_dict["gyro"][2]  # Just keep k value
    for key in ["accelerometer", "velocimeter"]:
        modified_state_dict[key] = modified_state_dict[key][:2]  # Remove z axis value
    stacked_env_state = np.hstack(list(modified_state_dict.values()))
    return np.hstack((robot_position, stacked_env_state))


def preprocess_state2(robot_state, hazard_vector):
    """Pre process state."""
    relative_hazards = hazard_vector - robot_state[:2]
    hazard_distances = np.linalg.norm(relative_hazards, axis=1)
    sorted_hazards = relative_hazards[np.argsort(-hazard_distances)]
    return np.hstack((robot_state, sorted_hazards.flatten()))


class MPCLearner:
    gamma = 0.99
    dropout, l2_penalty = 0, 0
    neurons, layers = 500, 2
    MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE = 100000, 10000  # Minimum number of steps in a memory to start training
    MINIBATCH_SIZE = 512

    def __init__(self, state_dims=10, action_dims=2):
        self.trajectory_length = 40
        self.no_trajectories = 100
        self.n_parents = 5
        self.REPLAY_MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.state_dims = 10
        self.hazard_dims = state_dims
        self.action_dims = action_dims
        self.combined_dims = self.state_dims + self.action_dims
        self.policy_dims = self.state_dims + self.hazard_dims
        self.best_trajectories = None

        self.dynamics_model = create_model(input_size=self.combined_dims,
                                           output_size=self.state_dims,
                                           output_activation="linear",
                                           neurons=self.neurons,
                                           layers=self.layers,
                                           dr=self.dropout,
                                           l2_penalty=self.l2_penalty)
        self.input_mean, self.input_std, self.output_mean, self.output_std = [0, 1, 0, 1]
        self.normalised = False
        self.safety_threshold = 0.8

    def model_prediction(self, robot_state, action):
        state_action = np.expand_dims(np.hstack((robot_state, action)), axis=1).T
        normalised_input = (state_action - self.input_mean) / self.input_std
        normalised_output = self.dynamics_model.predict(normalised_input)
        return normalised_output * self.output_std + self.output_mean

    def generate_from_parents(self, policy_actions):
        """Generate candidate child action sequences from parents of previous generation."""
        parent_trajectories = np.vstack((policy_actions, self.best_trajectories))
        child_trajectories = None

        automatic = True
        if automatic:
            sigmas = np.squeeze(np.clip(np.std(policy_actions, axis=1), 1e-2, 1))
            deltas = np.sum(np.abs(policy_actions[:, 1:, :] - policy_actions[:, :-1, :]), axis=1)
            length_scales = np.clip(np.squeeze(self.trajectory_length / (deltas / sigmas)), 1, self.trajectory_length)
        else:
            sigmas = np.array([0.01, 0.2])
            length_scales = np.array([20, 20])

        for i in range(self.n_parents):
            action_sequences = np.repeat(parent_trajectories[i:i+1,:,:], self.no_trajectories/self.n_parents, axis=0)
            child_trajectories = action_sequences if child_trajectories is None else \
                                                    np.vstack((child_trajectories, action_sequences))


        # print(sigmas, deltas, length_scales)
        correlated_noise = add_correlated_noise(sigmas, length_scales, independent=False)
        #child_trajectories += correlated_noise
        """
        for dim in range(self.action_dims):
            traj_cov = np.cov(parent_trajectories[:, :, dim], rowvar=False)
            child_dim = np.random.multivariate_normal(traj_mean[:, dim], traj_cov, size=no_trajectories)
            child_trajectories.append(child_dim)"""
        np.clip(child_trajectories+correlated_noise, -1, 1, out=child_trajectories)

        plot = True if random.random() > 1 else False

        if plot:
            plt.figure()
            fig, axes = plt.subplots(2, 1, sharex=True, squeeze=True)
            for i in range(2):
                for k in range(5):
                    axes[i].plot(parent_trajectories[k, :, i], "--", linewidth=0.5, color="green") #, label="Parent trajectories")
                for j in range(100):
                    axes[i].plot(child_trajectories[j, :, i], "--", linewidth=0.1, color="red") #, label="Sample trajectories")
            axes[0].set_ylabel("Forward action, f")
            axes[1].set_ylabel("Rotation action, r")
            axes[1].set_xlabel("Sample index")
            #plt.legend()
            # plt.savefig("correlatedpolicyimprovement")
            plt.show()
        return child_trajectories

    def generate_policy_trajectory(self, initial_robot_state, goal_pos, hazard_vector, policy):
        """Generate sequence of policy actions from current state."""
        current_robot_state = deepcopy(initial_robot_state)
        old_robot_state = current_robot_state
        policy_actions = []
        policy_reward = 0
        avg_policy_cost, max_policy_cost = 0, 0
        discount = 1
        for i in range(self.trajectory_length):
            terminal = i == self.trajectory_length-1
            if policy == human_policy:
                next_action = policy(np.squeeze(current_robot_state), goal_pos, hazard_vector)
            else:
                policy_state = preprocess_state2(current_robot_state, hazard_vector)
                next_action = policy(policy_state)
            policy_actions.append(next_action)
            delta_state = np.squeeze(self.model_prediction(current_robot_state, next_action))
            current_robot_state = old_robot_state + delta_state
            state_cost = compute_hazard_cost(current_robot_state, goal_pos, hazard_vector)
            max_policy_cost = max(max_policy_cost, state_cost)
            avg_policy_cost += state_cost
            policy_reward += discount * compute_value(current_robot_state, old_robot_state, goal_pos, hazard_vector, terminal)
            discount *= self.gamma
            old_robot_state = current_robot_state

        avg_policy_cost /= self.trajectory_length
        policy_actions = np.array(policy_actions).reshape(1, self.trajectory_length, self.action_dims)
        return policy_actions, avg_policy_cost, max_policy_cost, policy_reward

    def generate_from_policy(self, policy_actions):
        """Generate candidate action sequences from policy."""
        action_sequences = np.repeat(policy_actions, self.no_trajectories, axis=0)
        automatic = False
        if automatic:
            sigmas = np.squeeze(np.clip(np.std(policy_actions, axis=1), 1e-2, 1))
            deltas = np.sum(np.abs(policy_actions[:, 1:, :] - policy_actions[:, :-1, :]), axis=1)
            length_scales = np.clip(np.squeeze(self.trajectory_length / (deltas / sigmas)), 0.01, self.trajectory_length)
        else:
            sigmas = np.array([0.01, 0.2])
            length_scales = np.array([20, 20])

        # print(sigmas, deltas, length_scales)
        correlated_noise = add_correlated_noise(sigmas, length_scales, independent=False)
        mpc_action_sequences = np.clip(action_sequences + correlated_noise, -1, 1)
        return mpc_action_sequences

    def compute_trajectories(self, initial_robot_state, goal_pos, hazard_vector, policy):
        trajectory_length = 40  # MPC Horizon length
        no_trajectories = 100
        trajectory_values = np.zeros(no_trajectories)
        trajectory_max_costs = np.zeros(no_trajectories)
        trajectory_avg_costs = np.zeros(no_trajectories)
        trajectory_speed_costs = np.zeros(no_trajectories)
        current_states = np.repeat(initial_robot_state.reshape(1, self.state_dims), no_trajectories, axis=0)
        old_robot_states = current_states
        policy_actions, avg_policy_cost, max_policy_cost, policy_reward = self.generate_policy_trajectory(
            initial_robot_state, goal_pos, hazard_vector, policy)
        if self.best_trajectories is None:
            mpc_action_sequences = self.generate_from_policy(policy_actions)
        else:
            mpc_action_sequences = self.generate_from_parents(policy_actions)
        discount = 1
        for i in range(trajectory_length):
            state_actions = np.hstack((current_states, mpc_action_sequences[:, i]))
            normalised_inputs = (state_actions - self.input_mean) / self.input_std
            normalised_outputs = self.dynamics_model.predict(normalised_inputs)
            delta_states = normalised_outputs*self.output_std + self.output_mean
            current_states = old_robot_states + delta_states
            state_hazard_costs = np.apply_along_axis(compute_hazard_cost, 1, current_states, goal_pos, hazard_vector)
            trajectory_avg_costs += state_hazard_costs
            speed_costs = np.linalg.norm(current_states[:, 7:9], axis=1)
            trajectory_speed_costs = np.max((trajectory_speed_costs, speed_costs), axis=0)
            terminal = i == trajectory_length-1
            state_values = compute_values(current_states, old_robot_states, goal_pos, hazard_vector, terminal)
            trajectory_max_costs = np.max((trajectory_max_costs, state_hazard_costs), axis=0)  #  Infinite norm penalty on states
            trajectory_values += discount*state_values
            discount *= self.gamma
            old_robot_states = current_states
        trajectory_avg_costs /= trajectory_length

        avg_cost_limit = self.safety_threshold
        max_cost_limit = 0.85
        speed_limit = 1.5
        # Discount all trajectories less safe than limit.
        bad_trajectories1 = np.where(trajectory_avg_costs > avg_cost_limit)
        trajectory_values[bad_trajectories1] = -np.inf
        bad_trajectories2 = np.where(trajectory_max_costs > max_cost_limit)
        trajectory_values[bad_trajectories2] = -np.inf
        bad_trajectories3 = np.where(trajectory_speed_costs > speed_limit)
        trajectory_values[bad_trajectories3] = -np.inf
        best_trajectory = np.argmax(trajectory_values)
        self.best_trajectories = mpc_action_sequences[np.argsort(-trajectory_values)[:self.n_parents-1]]

        #print("{} safe trajectories found".format(len(np.where(trajectory_values != -np.inf)[0])))

        """if policy_reward > trajectory_values[best_trajectory]:
            print(r"Using $\pi_B$")
            if avg_policy_cost > self.safety_threshold or max_policy_cost > 0.85:
                print("Couldn't find any safe policy :(")
            return action_sequences[0, 0], 0"""
        if max_policy_cost < 0.85 and avg_policy_cost < avg_cost_limit:
            if policy_reward > np.max(trajectory_values):
                print("Policy is best")
                print(policy_actions.shape)
                return policy_actions[0, 0], 0

        if np.max(trajectory_values) == -np.inf:
            print("No safe trajectories found. Emergency stop.")
            """if avg_policy_cost > self.safety_threshold or max_policy_cost > 0.85:
                print("Base policy also unsafe :(")"""
            return np.array([0, 0]), 0
        else:
            #print("Best: ", mpc_action_sequences[best_trajectory, 0], trajectory_values[best_trajectory], trajectory_avg_costs[best_trajectory],
            #      trajectory_max_costs[best_trajectory])

            #print("Policy: ", policy_actions[0, 0], policy_reward, avg_policy_cost, max_policy_cost)

            if random.random() > 1:
                plt.figure()
                fig, axes = plt.subplots(2, 1, sharex=True, squeeze=True)
                for i in range(2):
                    for j in range(100):
                        if j == best_trajectory:
                            continue
                        axes[i].plot(mpc_action_sequences[j, :, i], "--", linewidth=0.1, color="red")
                    axes[i].plot(mpc_action_sequences[best_trajectory, :, i], "--", linewidth=0.2, color="red", label="Sample trajectories")
                    #axes[i].plot(policy_actions[0, :, i], color="blue", label="Base policy trajectory")
                    axes[i].plot(mpc_action_sequences[best_trajectory, :, i], color="green", label="Best sample trajectory")
                axes[0].set_ylabel("Forward action, f")
                axes[1].set_ylabel("Rotation action, r")
                axes[1].set_xlabel("Sample index")
                plt.legend()
                #plt.savefig("correlatedpolicyimprovement")
                plt.show()
                #raise ValueError
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


def add_correlated_noise(sigmas=[0.1], lengthscales=[0.1], trajectory_length=40, no_trajectories=100, independent=True):
    """Add exploration noise to action sequences of each dimension."""
    mean = trajectory_length * [0]
    array_inds = np.arange(0, trajectory_length).reshape(trajectory_length, 1)
    correlations = []
    for (sigma, lengthscale) in zip(sigmas, lengthscales):
        if independent:
            correlated_noise = np.random.normal(0, sigma, size=(no_trajectories, trajectory_length))
        else:
            cov_matrix = (sigma ** 2) * np.exp(-((array_inds.T - array_inds) / lengthscale) ** 2)
            correlated_noise = np.random.multivariate_normal(mean, cov_matrix, size=no_trajectories)
        correlations.append(correlated_noise)

    correlations = np.moveaxis(np.array(correlations), 0, -1)
    #print(correlations.shape)
    return correlations


def compute_hazard_cost(robot_state, goal_pos, hazards):
    sigma = 0.3
    displacements = hazards - (robot_state[:2]+goal_pos)
    distances = np.einsum('ij,ji->i', displacements, displacements.T)
    speed_penalty = np.exp(-np.linalg.norm(robot_state[7:9]))
    hazard_rating = np.max(np.exp(-(distances / sigma)**2))
    return hazard_rating


def compute_speed_cost(robot_state):
    """Linear penalty on speed."""
    return np.linalg.norm(robot_state[7:9])


def compute_value(robot_state, old_robot_state, goal_pos, hazards, terminal=False, print_reward=False):
    distance_reward = 10*(np.linalg.norm(old_robot_state[:2]) - np.linalg.norm(robot_state[:2]))
    end_reward = 10 if np.linalg.norm(robot_state[:2]) < 0.2 else 0
    distance_reward2 = robot_state[2] if terminal else 0
    hazard_penalty = compute_hazard_cost(robot_state, goal_pos, hazards) if terminal else 0
    #angle_reward = robot_state[3]*(1-hazard_penalty)  # Encourage aiming for goal away from hazards
    if print_reward:
        print("DEHA", distance_reward, end_reward, hazard_penalty, distance_reward2)
    return distance_reward+end_reward-hazard_penalty+distance_reward2 #+angle_reward #-angle_penalty


def compute_values(robot_states, old_robot_states, goal_pos, hazards, terminal=False):
    return np.array([compute_value(robot_state, old_robot_state, goal_pos, hazards, terminal)
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

    for i in range(EPISODES):
        no_accepted = 0
        stored, total, episode_reward, episode_cost, discount = 0, 0, 0, 0, 1
        gamma = 0.99
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
            new_env_state, env_reward, done, info = env.step(action)

            episode_cost += info["cost"]
            new_position = env.robot_pos[:2] - env.goal_pos[:2]
            new_robot_state = form_state(new_env_state, new_position)

            episode_reward += discount*compute_value(new_robot_state, robot_state, goal_pos, hazard_vector)
            discount *= gamma
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
        #print(f"Buffer length: {len(mpc_learner.REPLAY_MEMORY)}")
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
        loaded_learner = pickle.load(open("MPCModels/mpcmodel1", "rb"))
        seeds = range(35, 48)
        for seed in seeds:
            env.seed(seed)  # 18 good
            #np.random.seed(100)
            main(EPISODES=1, mpc_learner=loaded_learner, render=True, policy=human_policy, save_name=None)
