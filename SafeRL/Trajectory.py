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
from MPC import form_state
from Controller import human_policy
from matplotlib import pyplot as plt


def main(sequence="action_sequence"):
    reset_seeds()
    action_sequence = pickle.load(open(f"Trajectories/{sequence}", "rb"))
    episode_reward, episode_cost = 0, 0
    env_state = env.reset()

    position = env.robot_pos[:2] - env.goal_pos[:2]
    robot_state = form_state(env_state, position)
    env.render()
    done = False
    steps = 0
    while not done:
        action = action_sequence.pop(0)
        action_sequence.append(action)
        new_env_state, reward, done, info = env.step(action)
        episode_reward += reward
        episode_cost += info["cost"]
        steps += 1
        env.render()

    print(f"Steps {steps}")
    return

def reset_seeds(seed=18):
    env.seed(seed)  # 18 good
    np.random.seed(seed)

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
    env.seed(18)  # 18 good
    np.random.seed(1)
    opt_seq = np.array(pickle.load(open("Trajectories/action_sequence2", "rb")))
    base_seq = np.array(pickle.load(open("Trajectories/base_action_sequence", "rb")))
    print(opt_seq[:, 0].shape)
    print(base_seq[:,0].shape)
    fig, axes = plt.subplots(2,1)
    ax = axes[0]
    ax.plot(base_seq[:, 0], color="red")
    ax.plot(opt_seq[:, 0], color="blue")
    ax = axes[1]
    ax.plot(base_seq[:, 1], color="red")
    ax.plot(opt_seq[:, 1], color="blue")
    plt.show()
    main("base_action_sequence")
    main("action_sequence2")
