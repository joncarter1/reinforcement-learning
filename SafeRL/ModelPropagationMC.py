import tensorflow as tf
import gym
import random
import numpy as np
import pickle
from tqdm import tqdm
from keras.models import load_model
import os
from copy import deepcopy
from collections import deque
from SafeRL.ModelLearning import gaussian_nll
from SafeRL.MPCLearner import MPCLearner2, zero_func
import matplotlib.pyplot as plt
import keras.losses
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"gaussian_nll": gaussian_nll})


def main(EPISODES=1, mpc_learner=None, no_particles=20, render=True, save=False):
    if mpc_learner is None:
        mpc_learner = MPCLearner2()

    for i in tqdm(range(EPISODES)):
        state = env.reset()
        predicted_states = np.repeat(state.reshape(1, mpc_learner.state_dims), no_particles, axis=0)
        prediction_sigmas = np.zeros(shape=predicted_states.shape)
        true_path = [np.squeeze(state)]
        trajectories = [deepcopy(predicted_states)]
        trajectory_sigmas = [np.zeros(shape=predicted_states.shape)]

        if render:
            env.render()
        done = False
        i = 0

        while not done:
            action = np.random.uniform(-1, 1, size=(1,))
            #action = np.array([1]) if ((i//50)%2) else np.array([-1])
            i += 1

            for p_i, predicted_state in enumerate(predicted_states):
                b_i = p_i #//5  #np.random.randint(0, 5)
                model_output = mpc_learner.model_prediction(predicted_state, action, b_i)
                if mpc_learner.probabilistic:
                    mean, sigma = model_output
                    prediction_sigmas[p_i] = np.clip(sigma, mpc_learner.min_sigmas[b_i], mpc_learner.max_sigmas[b_i])
                    cov = np.diag(np.squeeze(sigma ** 2))
                else:
                    mean = model_output

                try:
                    new_predicted_state = predicted_state + mean  #+ np.random.multivariate_normal(np.squeeze(mean), 0*cov)
                except Exception as e:
                    print(mean, cov, predicted_state)
                    raise e
                predicted_states[p_i] = new_predicted_state

            state, reward, done, info = env.step(action)
            true_path.append(deepcopy(state))
            trajectories.append(deepcopy(predicted_states))
            trajectory_sigmas.append(deepcopy(prediction_sigmas))
            if render:
                env.render()

    #trajectories = [np.array(entry) for entry in trajectories]
    return np.array(true_path), np.array(trajectories), np.array(trajectory_sigmas)


def soft(x):
    x_min, x_max = -10, 10
    x = x_max - np.log(np.exp(x_max-x)+1)
    x = x_min + np.log(np.exp(x-x_min)+1)
    return x


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    loaded_learner = pickle.load(open("MountainCarPE", "rb"))
    no_particles = 5
    true_path, trajectories, trajectory_sigmas = main(EPISODES=1, mpc_learner=loaded_learner, no_particles=no_particles,
                                                         render=False, save=False)
    data_length = trajectories.shape[0]
    mean_vector = np.mean(trajectories, axis=1)
    std_vector = np.std(trajectories, axis=1)
    max_vector = np.max(trajectories, axis=1)
    min_vector = np.min(trajectories, axis=1)

    max_steps = 200
    plt.ion()
    plt.figure()
    fig, axes = plt.subplots(2, 1)
    for k in range(2):
        axes[k].plot(true_path[:, k], label="Truth")
        axes[k].fill_between(np.arange(0, data_length), true_path[:, k], true_path[:, k], alpha=0.5)
        axes[k].set_xlim(0, max_steps)
        axes[k].set_ylim(np.min(true_path[:, k])-0.5*np.abs(np.min(true_path[:, k])), np.max(true_path[:, k])+0.5*np.abs(np.max(true_path[:, k])))
        #axes[k].plot(mean_vector[:, k], label="Prediction mean", color="tab:red")
        #std_lower = mean_vector[:, k] - 3*std_vector[:, k]
        #std_upper = mean_vector[:, k] + 3*std_vector[:, k]
        #axes[k].fill_between(np.arange(0, data_length), min_vector[:, k], max_vector[:, k], color="tab:red", alpha=0.5)
        for p_i in range(5):
            axes[k].plot(trajectories[:, p_i, k])
            std_vector = trajectory_sigmas[:, p_i, k]
            #print(std_vector)
            std_lower = trajectories[:, p_i, k] - std_vector
            std_upper = trajectories[:, p_i, k] + std_vector
            axes[k].fill_between(np.arange(0, data_length), std_lower, std_upper, alpha=0.5)
    plt.legend()
    plt.show()