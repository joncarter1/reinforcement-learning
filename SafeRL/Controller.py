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
from ModelLearning import NNModel
import matplotlib.pyplot as plt
"""
-ve rotation = clockwise torque
i.e. theta defined in the polar sense.
"""

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


default_dist = 2


def compute_cost(position, hazards):
    sigma = 0.3
    displacements = hazards - position
    distances = np.einsum('ij,ji->i', displacements, displacements.T)
    return np.sum(np.exp(-distances / (sigma ** 2)))


def path_finder(robot_state, goal_pos, hazards):
    lookahead_dist = 0.4
    directions = 501
    goal_ind = (directions//2)
    sigma = 0.3
    sigma_a = np.pi
    safety_threshold = 0.3
    angles = np.linspace(-np.pi/2, np.pi/2, directions)
    xs = np.stack((np.sin(angles), np.cos(angles)))*lookahead_dist
    displacement_vector = -robot_state[:2]  # Vector from robot to goal
    displacement_vector /= np.linalg.norm(displacement_vector)
    sin, cos = displacement_vector
    rotation_matrix = np.array([[cos, sin], [-sin, cos]])
    robot_pos = (goal_pos+robot_state[:2]).reshape((1, 2, 1))
    trajectories = robot_pos + np.expand_dims(rotation_matrix@xs, axis=0)
    hazards = np.expand_dims(hazards, axis=-1)
    stacked_hazards = np.repeat(hazards, directions, axis=2)
    displacements = stacked_hazards-trajectories
    sq_distances = np.einsum('ij...,ji...->i...', displacements, np.swapaxes(displacements, 0, 1))
    costs = np.sum(np.exp(-sq_distances/(sigma**2)), axis=0)
    angle_penalties = 1-np.exp(-angles**2/(sigma_a**2))
    l2 = 1
    angle_values = angle_penalties+l2*costs
    path_ind = np.argmin(angle_values)
    delta_angle = angles[path_ind]*180/np.pi
    cost = costs[path_ind]
    return delta_angle, cost


REPLAY_MEMORY = []


def store_transition(robot_state, action, hazards, new_robot_state):
    stacked_transition = np.hstack((robot_state, action, hazards, new_robot_state))
    REPLAY_MEMORY.append(stacked_transition)
    return


def save_results(memory_buffer):
    pickle.dump(memory_buffer, open(f"policy_data/replay_memory", "wb"))
    return True


def get_forward(path_cost):
    return 0.01*(np.exp(-5*path_cost))


def safe_policy(robot_state, goal_pos, hazard_array):
    velocity = np.linalg.norm(robot_state[7:9])
    angular_velocity = robot_state[-1]
    k1, k2 = 0, 0
    return np.array([-k1*velocity, -k2*angular_velocity])


def human_policy(robot_state, goal_pos, hazard_array):
    """Hand crafted controller for problem."""
    cos_theta, sin_theta = np.clip(robot_state[3:5], -1, 1) #goal_compass
    goal_angle = angle_invert(sin_theta, cos_theta)*180/np.pi
    delta_angle, cost = path_finder(robot_state, goal_pos, hazard_array)
    new_target_angle = goal_angle - delta_angle
    forward = get_forward(cost)
    k = 0.1
    rotation = sigmoid(k*new_target_angle)
    if np.isnan(rotation) or np.isnan(forward):
        print(robot_state)
        print(sin_theta)
        print(cos_theta)
        raise ValueError
    return np.array([forward, rotation])


def modify_state(state_dict):
    modified_state_dict = deepcopy(state_dict)
    del modified_state_dict["magnetometer"]  # Useless state measurement.
    del modified_state_dict["gyro"]  # Useless state measurement.
    #modified_state_dict["gyro"] = modified_state_dict["gyro"][2]  # Just keep k value
    for key in ["accelerometer", "velocimeter"]:
        modified_state_dict[key] = modified_state_dict[key][:2]  # Remove z axis value
    print(state_dict)
    return modified_state_dict


def main(EPISODES, render=False, save=False, policy=human_policy):
    ep_rewards = []
    ep_costs = []
    actions = []

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
            if np.random.random() < 0.3:
                action = np.random.uniform(-1, 1, size=(1,2))
            else:
                action = policy(state, env_state, env).reshape(1, 2)
            actions.append(action.T)
            print(state)
            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_cost += info["cost"]
            risky_state = np.min(-np.log(new_state["hazards_lidar"])) < 0.5
            if save:
                if risky_state or np.random.random() < 0.1:
                    stored += 1
                if np.random.random() < 0.3:
                    learner.train_models()
            total += 1
            state = new_state

            if render:
                env.render()

        print(f"Episode {i}, reward {episode_reward}, cost {episode_cost}, stored {stored}/{total}")
        if save:
            print(f"Buffer size: {len(learner.REPLAY_MEMORY)}")
        ep_rewards.append(episode_reward)
        ep_costs.append(episode_cost)

        if save and i != 0 and not i%(EPISODES//10):
            learner.save("jointmodel")

    if save:
        learner.save("jointmodel")
    return actions

lidar_bins = 32


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
        'observe_circle': True,
        'observe_vision': False,
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
    #loaded_learner = pickle.load(open("jointmodel", "rb"))
    actions = main(EPISODES=10, render=True, policy=human_policy, save=False)
    plt.figure()
    actions = np.array(actions)
    print(actions.shape)
    plt.scatter(actions[:, 0], actions[:, 1])
    plt.xlabel("Forward")
    plt.ylabel("Rotation")
    plt.show()