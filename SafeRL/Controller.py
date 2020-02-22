import safety_gym
import gym
from gym.utils.play import play
import pandas as pd
import time
import threading
import numpy as np
import pickle
from safety_gym.envs.engine import Engine
from tqdm import tqdm
from keras.models import load_model

"""
-ve rotation = clockwise torque
i.e. theta defined in the polar sense.
"""

lidar_bins = 32
safety_threshold1 = 0.75
safety_threshold2 = 0.7

try:
    state_action_buffer = list(pickle.load(open(f"policy_data/state_action_buffer", "rb")))
    delta_state_buffer = list(pickle.load(open(f"policy_data/delta_state_buffer", "rb")))
    print(len(state_action_buffer))
except FileNotFoundError:
    state_action_buffer = []
    delta_state_buffer = []

model = load_model("model1")


def sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1

def sat_exp(x, k):
    """Saturating exponential function."""
    return 1-np.exp(-k*x)

def nn_policy(state):
    stacked_state = np.expand_dims(np.hstack(list(state.values())), axis=1).T
    return model.predict(x=stacked_state)


def angle_invert(sin_angle, cos_angle):
    return np.arccos(cos_angle)*np.sign(sin_angle)


def lidar_mapping(angle):
    """Get index of lidar closest to a given angle."""
    conv_angle = angle % 360  # Convert to 0 - 360 range.
    angular_segment = 360/lidar_bins
    index = int(np.round(conv_angle / angular_segment) % lidar_bins)
    return index


def path_finder(hazard_lidar, goal_ind):
    safe_path_found = False  # Found path where 3 adjacent lidar readings not proximal.
    delta = 0
    best_safety = None  # Safest path direction found
    forward_range = np.arange(-4, 5)
    safe_directions = np.where(hazard_lidar < safety_threshold2)[0]
    sort_key = lambda x : min(abs(x-goal_ind), lidar_bins - abs(x-goal_ind))  # Sort safe directions by distance from goal ind.
    sorted_safe_directions = sorted(safe_directions, key=sort_key)

    for direction in sorted_safe_directions:
        safety1 = np.max(hazard_lidar[(direction+forward_range)%lidar_bins])
        if safety1 < safety_threshold1:
            return direction, 9
        safety2 = 1
        i = 1
        direction_safe = True

        """Expand radar around direction summing number of safe adjacent directions."""
        while direction_safe and i < 5:
            for j in [i, -i]:
                if (direction+i)%lidar_bins in sorted_safe_directions or direction+i in sorted_safe_directions:
                    safety2 += 1
                else:
                    direction_safe = False
            i += 1

        if best_safety is None or safety2 > best_safety:
            best_safety = safety2
            best_direction = direction

    return best_direction, best_safety



def get_forward(hazards_lidar):
    lidar_range = np.arange(-lidar_bins//4, lidar_bins//4 + 1)
    hazard_factor = 1/np.max(hazards_lidar[lidar_range])
    k1 = 0.01
    forward = sat_exp(hazard_factor, k1)
    return forward


def human_policy(new_state):
    """Hand crafted controller for problem."""
    goal_dist = 1 / new_state["goal_dist"]
    goal_compass = new_state["goal_compass"]
    cos_theta, sin_theta = goal_compass
    goal_angle = angle_invert(sin_theta, cos_theta)*180/np.pi
    lidar_range = np.arange(-lidar_bins//4, lidar_bins//4 + 1)
    goal_ind = lidar_mapping(goal_angle)

    #print("Goal direction/angle", goal_ind, goal_angle)
    #grem_lidar = 1/new_state["gremlins_lidar"]
    hazards_lidar = new_state["hazards_lidar"]
    #print("Hazards:", hazards_lidar)
    forward_hazards = hazards_lidar[(lidar_range+goal_ind)%lidar_bins]
    #print("Goal direction hazards", forward_hazards)
    #forward_gremlins = grem_lidar[(lidar_range+goal_ind)%16]

    path_ind, path_safety = path_finder(hazards_lidar, goal_ind)
    #print("Path",  path_ind, path_safety)
    target_angle = (360/lidar_bins)*path_ind
    if target_angle > 180:
        target_angle = target_angle - 360
    #print(grem_lidar[lidar_range + lidar_ind])
    forward = get_forward(hazards_lidar)
    k = 0.1
    rotation = sigmoid(k*target_angle)
    #print("Forward", forward)
    #print("Rotation", rotation)
    return np.array([forward, rotation])


def store_transition(state, action, new_state):
    stacked_state = np.hstack(list(state.values()))
    new_stacked_state = np.hstack(list(new_state.values()))
    state_action_vector = np.hstack((stacked_state, action))

    delta_state_vector = new_stacked_state - stacked_state
    state_action_buffer.append(state_action_vector)
    delta_state_buffer.append(delta_state_vector)


def main(EPISODES, render=False, save=False, policy=human_policy):
    ep_rewards = []
    ep_costs = []

    for i in tqdm(range(EPISODES)):
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
            """print("Goal", 1/state["goal_lidar"])
            print("Hazards", 1/state["hazards_lidar"])
            print("Gremlins", 1/state["gremlins_lidar"])"""
            if save:
                store_transition(state, action, new_state)
            state = new_state

            if render:
                env.render()

        print(f"Episode {i}, reward {episode_reward}, cost {episode_cost}")
        ep_rewards.append(episode_reward)
        ep_costs.append(episode_cost)

    if save:
        ep_stats = {"rewards":ep_rewards, "costs":ep_costs}
        pickle.dump(np.array(state_action_buffer), open(f"policy_data/state_action_buffer", "wb"))
        pickle.dump(np.array(delta_state_buffer), open(f"policy_data/delta_state_buffer", "wb"))
        pickle.dump(ep_stats, open(f"policy_data/episode_stats", "wb"))
    return


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
        'lidar_max_dist': 3,
        'lidar_num_bins': lidar_bins,
        'hazards_num': 6,
        'vases_num': 0,
        'gremlins_num': 0,
        'gremlins_travel': 0.5,
        'gremlins_keepout': 0.4,
    }
    env = Engine(config)
    main(EPISODES=100, render=False, policy=human_policy, save=True)
