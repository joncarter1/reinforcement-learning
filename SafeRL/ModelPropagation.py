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
from Controller import human_policy
import matplotlib.pyplot as plt


def main(EPISODES, mpc_learner=None, render=False, save=False, policy=human_policy):
    if mpc_learner is None:
        mpc_learner = MPCLearner()

    for i in tqdm(range(EPISODES)):
        env_state = env.reset()
        position = env.robot_pos[:2]-env.goal_pos[:2]
        robot_state = form_state(env_state, position)
        predicted_state = robot_state
        true_path = [np.squeeze(robot_state)]
        trajectories = [[np.squeeze(predicted_state)] for _ in range(10)]
        trajectory = [np.squeeze(predicted_state)]
        if render:
            env.render()
        done = False
        while not done:
            if np.random.random() < 0:
                action = np.random.uniform(-1, 1, size=(2,))
            else:
                hazards = env.gremlins_obj_pos + env.hazards_pos
                action = policy(robot_state, env.goal_pos[:2], np.array(hazards)[:, :2]).reshape(2, )

            #action = np.random.random(size=(2,)) - 0.5
            print(env_state["goal_dist"])
            new_predicted_state = np.squeeze(predicted_state + mpc_learner.model_prediction(predicted_state, action))
            new_env_state, reward, done, info = env.step(action)
            new_position = env.robot_pos[:2]-env.goal_pos[:2]

            new_robot_state = form_state(new_env_state, new_position)
            hazards = np.array(env.hazards_pos)[:, :2]
            env_state = new_env_state
            robot_state = new_robot_state
            predicted_state = new_predicted_state
            true_path.append(deepcopy(robot_state))
            trajectory.append(np.squeeze(predicted_state))
            if render:
                env.render()

    trajectories = [np.array(entry) for entry in trajectories]
    return np.array(true_path), np.array(trajectory), trajectories


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
    print(len(loaded_learner.REPLAY_MEMORY))
    true_path, trajectory, trajectories = main(EPISODES=1, mpc_learner=loaded_learner, render=True, policy=human_policy, save=False)

    max_steps = 10000
    ind1, ind2 = 0, 1
    state_items = ["x", "y", "Dist", "cos", " sin"]
    plt.figure()
    fig, axes = plt.subplots(5,2)
    for k in range(10):
        i = k%5
        j = k//5
        axes[i, j].plot(true_path[:max_steps, k], label="Truth")
        axes[i, j].plot(trajectory[:max_steps, k], label="Prediction")
        axes[i, j].set_xlabel("Timesteps")
        #plt.legend()
    plt.show()

    plt.figure()
    #plt.xlim(-4, 4)
    #plt.ylim(-4, 4)
    plt.scatter(true_path[:max_steps, ind1], true_path[:max_steps, ind2], marker=".", s=15, label="Ground truth")
    plt.scatter(trajectory[:max_steps, ind1], trajectory[:max_steps, ind2], color="red", marker=".", s=15, label="Model prediction")

    T_steps = [0.5, 1, 2]
    for t in T_steps:
        x = int(50*t)
        plt.scatter(true_path[x, ind1], true_path[x, ind2], marker="x", s=50, label=f"T = {t}s")
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.legend()
    plt.show()