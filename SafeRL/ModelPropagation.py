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
from MPC import MPCLearner, form_state
from Controller import human_policy, Learner
import matplotlib.pyplot as plt


def main(EPISODES, mpc_learner=None, render=False, save=False, policy=human_policy):
    if mpc_learner is None:
        mpc_learner = MPCLearner()

    for i in tqdm(range(EPISODES)):
        env_state = env.reset()
        position = env.robot_pos[:2]
        robot_state = form_state(env_state, position)
        predicted_state = robot_state
        true_path = [deepcopy(robot_state)]
        trajectory = [deepcopy(predicted_state)]
        if render:
            env.render()
        done = False
        while not done:
            if policy == human_policy:
                action = human_policy(env_state)
            else:
                hazards = np.array(env.hazards_pos)[:, :2]
                flat_hazard_vector = deepcopy(hazards).flatten()
                action = policy(robot_state, flat_hazard_vector)
            new_predicted_state = predicted_state + mpc_learner.model_prediction(robot_state, action)
            new_env_state, reward, done, info = env.step(action)
            new_position = env.robot_pos[:2]

            new_robot_state = form_state(new_env_state, new_position)
            hazards = np.array(env.hazards_pos)[:, :2]
            flat_hazard_vector = deepcopy(hazards).flatten()

            env_state = new_env_state
            robot_state = new_robot_state
            predicted_state = new_predicted_state
            true_path.append(deepcopy(robot_state))
            trajectory.append(np.squeeze(predicted_state))
            if render:
                env.render()

    return np.array(true_path), np.array(trajectory)


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

    loaded_learner = pickle.load(open("mpcmodel", "rb"))
    true_path, trajectory = main(EPISODES=1, mpc_learner=loaded_learner, render=True, policy=loaded_learner, save=False)
    print(true_path.shape)
    print(trajectory.shape)
    plt.figure()
    plt.plot(true_path[:,8])
    plt.plot(trajectory[:,8])
    plt.show()