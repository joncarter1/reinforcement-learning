import safety_gym
import gym
import random
import numpy as np
import pickle
from safety_gym.envs.engine import Engine
from tqdm import tqdm
from keras.models import load_model
import os
from copy import deepcopy
from collections import deque
from SafeRL.ModelLearning import NNModel
"""
-ve rotation = clockwise torque
i.e. theta defined in the polar sense.
"""

class Learner:
    def __init__(self):
        self.MEMORY_SIZE = 50000
        self.MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
        self.REPLAY_MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.MINIBATCH_SIZE = 32
        self.state_dims = 39
        self.action_dims = 2
        self.combined_dims = self.state_dims+self.action_dims
        self.dynamics_model = NNModel(self.combined_dims, self.state_dims, "linear")
        self.policy = NNModel(self.state_dims, self.action_dims, "tanh")

    def __call__(self, state):
        modified_state = modify_state(state)
        stacked_state = np.expand_dims(np.hstack(list(modified_state.values())), axis=1).T
        return self.policy.predict(stacked_state)

    def store_transition(self, state, action, new_state):
        state_copy = modify_state(state)
        new_state_copy = modify_state(new_state)
        stacked_state = np.hstack(list(state_copy.values()))
        new_stacked_state = np.hstack(list(new_state_copy.values()))
        stacked_transition = np.hstack((stacked_state, action, new_stacked_state))
        self.REPLAY_MEMORY.append(stacked_transition)
        return

    def train_models(self):
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = np.array(random.sample(self.REPLAY_MEMORY, self.MINIBATCH_SIZE))

        self.dynamics_model.train(x=minibatch[:, :self.combined_dims],
                                y=minibatch[:, self.combined_dims:],
                                batch_size=self.MINIBATCH_SIZE)

        self.policy.train(x=minibatch[:, :self.state_dims],
                        y=minibatch[:, self.state_dims:self.combined_dims],
                        batch_size=self.MINIBATCH_SIZE)
        return

    def save(self, model_name):
        pickle.dump(self, open(model_name, "wb"))

def sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1


def sat_exp(x, k):
    """Saturating exponential function."""
    return np.exp(-k*x)


def nn_policy(state, nn_model):
    modified_state = modify_state(state)
    stacked_state = np.expand_dims(np.hstack(list(modified_state.values())), axis=1).T
    return nn_model.predict(stacked_state)


def angle_invert(sin_angle, cos_angle):
    return np.arccos(cos_angle)*np.sign(sin_angle)


def lidar_mapping(angle):
    """Get index of lidar closest to a given angle."""
    conv_angle = angle % 360  # Convert to 0 - 360 range.
    angular_segment = 360/lidar_bins
    index = int(np.round(conv_angle / angular_segment) % lidar_bins)
    return index


default_dist = 2


def path_finder(hazard_lidar, goal_ind):
    safety_threshold = 0.5
    best_safety = None  # Safest path direction found
    forward_range = np.arange(-4, 5)
    linear_lidar = -np.log(hazard_lidar)
    linear_lidar[linear_lidar > default_dist] = default_dist
    safe_directions = np.where(linear_lidar > safety_threshold)[0]
    sort_key = lambda x: min(abs(x-goal_ind), lidar_bins - abs(x-goal_ind))  # Sort safe directions by angle to goal.
    sorted_safe_directions = sorted(safe_directions, key=sort_key)


    for direction in sorted_safe_directions:
        safety2 = 1
        i = 1
        direction_safe = True

        # Expand radar around direction summing number of safe adjacent directions.
        while direction_safe and i < 7:
            for j in [i, -i]:
                if (direction + j) % lidar_bins in sorted_safe_directions or direction + j in sorted_safe_directions:
                    safety2 += 1
                else:
                    direction_safe = False
            i += 1

        if best_safety is None or safety2 > best_safety:
            best_safety = safety2
            best_direction = direction

    return best_direction, best_safety


def save_results(state_buffer, state_change_buffer, stats):
    pickle.dump(state_buffer, open(f"policy_data/state_action_buffer", "wb"))
    pickle.dump(state_change_buffer, open(f"policy_data/delta_state_buffer", "wb"))
    pickle.dump(stats, open(f"policy_data/episode_stats", "wb"))
    return True


def get_forward(cos_theta, path_ind, hazards_lidar):
    lidar_range = np.arange(-lidar_bins//4, lidar_bins//4 + 1)
    forward_hazards = hazards_lidar[(lidar_range+path_ind)%lidar_bins]
    forward = 0.5*cos_theta/(10+20*np.sum(hazards_lidar[(lidar_range+path_ind)%lidar_bins]))
    return forward


def human_policy(new_state):
    """Hand crafted controller for problem."""
    goal_compass = new_state["goal_compass"]
    cos_theta, sin_theta = goal_compass
    goal_angle = angle_invert(sin_theta, cos_theta)*180/np.pi
    goal_ind = lidar_mapping(goal_angle)

    hazards_lidar = new_state["hazards_lidar"]
    path_ind, path_safety = path_finder(hazards_lidar, goal_ind)
    target_angle = (360/lidar_bins)*path_ind
    if target_angle > 180:
        target_angle = target_angle - 360
    forward = get_forward(cos_theta, path_ind, hazards_lidar)
    k = 0.1
    rotation = sigmoid(k*target_angle)
    return np.array([forward, rotation])


def modify_state(state_dict):
    modified_state_dict = deepcopy(state_dict)
    del modified_state_dict["magnetometer"]  # Useless state measurement.
    del modified_state_dict["gyro"]  # Useless state measurement.
    for key in ["accelerometer", "velocimeter"]:
        modified_state_dict[key] = modified_state_dict[key][:2]  # Remove z axis value
    return modified_state_dict



def main(EPISODES, render=False, save=False, policy=human_policy):
    learner = Learner()
    ep_rewards = []
    ep_costs = []

    for i in tqdm(range(EPISODES)):
        stored = 0
        total = 0
        episode_reward = 0
        episode_cost = 0
        state = env.reset()
        if render:
            env.render()

        done = False
        while not done:
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_cost += info["cost"]
            risky_state = np.min(-np.log(new_state["hazards_lidar"])) < 0.5
            if save:
                if risky_state or np.random.random() < 0.1:
                    learner.store_transition(state, action, new_state)
                    stored += 1
                if np.random.random() < 0.3:
                    learner.train_models()
            total += 1
            state = new_state

            if render:
                env.render()

        print(f"Episode {i}, reward {episode_reward}, cost {episode_cost}, stored {stored}/{total}")
        ep_rewards.append(episode_reward)
        ep_costs.append(episode_cost)

        if save and i != 0 and not i%(EPISODES//10):
            learner.save("jointmodel")

    if save:
        learner.save("jointmodel")
    return

lidar_bins = 32
lidar_factor = 1


if __name__ == "__main__":
    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observation_flatten': False,
        'observe_goal_dist': True,
        'observe_goal_comp': True,
        'observe_goal_lidar': False,
        'observe_hazards': True,
        'observe_vases': False,
        'observe_gremlins': False,
        'constrain_hazards': True,
        'constrain_vases': False,
        'constrain_gremlins': False,
        'lidar_max_dist': None,
        'lidar_num_bins': lidar_bins,
        'hazards_num': 10,
        'vases_num': 0,
        'gremlins_num': 0,
        'gremlins_travel': 0.5,
        'gremlins_keepout': 0.4,
    }
    env = Engine(config)
    #learnt_policy = pickle.load(open("jointmodel", "rb"))
    main(EPISODES=100, render=False, policy=human_policy, save=True)