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
from Controller import human_policy, Learner
import matplotlib.pyplot as plt
"""
-ve rotation = clockwise torque
i.e. theta defined in the polar sense.
"""

class MPCLearner:
    def __init__(self):
        self.MEMORY_SIZE = 30000
        self.MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
        self.REPLAY_MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.MINIBATCH_SIZE = 32
        self.state_dims = 10
        self.no_hazards = 10
        self.action_dims = 2
        self.combined_dims = self.state_dims+self.action_dims
        self.dynamics_model = NNModel(self.combined_dims, self.state_dims, "linear")
        self.policy_dims = self.state_dims + self.no_hazards*2
        self.policy = NNModel(self.policy_dims, self.action_dims, "tanh")

    def __call__(self, robot_state, hazard_vector):
        input_vector = np.expand_dims(np.hstack((robot_state, hazard_vector)), axis=1).T
        return np.squeeze(self.policy.predict(input_vector))

    def model_prediction(self, robot_state, action):
        input_vector = np.expand_dims(np.hstack((robot_state, action)), axis=1).T
        return self.dynamics_model.predict(input_vector)

    def store_transition(self, robot_state, action, hazards, new_robot_state):
        delta_state = new_robot_state - robot_state  # Store change in state
        stacked_transition = np.hstack((robot_state, hazards, action, delta_state))
        self.REPLAY_MEMORY.append(stacked_transition)
        return

    def train_models(self):
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = np.array(random.sample(self.REPLAY_MEMORY, self.MINIBATCH_SIZE))
        policy_inds = np.r_[:self.state_dims, self.combined_dims:self.combined_dims + self.no_hazards*2]

        self.dynamics_model.train(x=minibatch[:, :self.combined_dims],
                                y=minibatch[:, -self.state_dims:],
                                batch_size=self.MINIBATCH_SIZE)

        self.policy.train(x=minibatch[:, policy_inds],
                        y=minibatch[:, self.state_dims:self.combined_dims],
                        batch_size=self.MINIBATCH_SIZE)
        return

    def save(self, model_name):
        pickle.dump(self, open(model_name, "wb"))



def compute_cost(position, hazards):
    sigma = 0.3
    position = env.robot_pos[:2]
    displacements = hazards-position
    distances = np.einsum('ij,ji->i', displacements, displacements.T)
    return np.sum(np.exp(-distances/(sigma**2)))


def form_state(state_dict, position):
    modified_state_dict = deepcopy(state_dict)
    for key in ["magnetometer", "hazards_lidar"]:
        del modified_state_dict[key]  # Useless state measurement.
    modified_state_dict["gyro"] = modified_state_dict["gyro"][2]  # Just keep k value
    for key in ["accelerometer", "velocimeter"]:
        modified_state_dict[key] = modified_state_dict[key][:2]  # Remove z axis value

    print(modified_state_dict)
    stacked_state = np.hstack(list(modified_state_dict.values()))
    return np.hstack((position, stacked_state))


def main(EPISODES, mpc_learner=None, render=False, save=False, policy=human_policy):
    if mpc_learner is None:
        mpc_learner = MPCLearner()

    for i in tqdm(range(EPISODES)):
        stored, total, episode_reward, episode_cost = 0, 0, 0, 0
        env_state = env.reset()
        position = env.robot_pos[:2]
        robot_state = form_state(env_state, position)
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
            new_env_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_cost += info["cost"]
            new_position = env.robot_pos[:2]
            new_robot_state = form_state(new_env_state, new_position)
            hazards = np.array(env.hazards_pos)[:, :2]
            flat_hazard_vector = deepcopy(hazards).flatten()

            if save and (compute_cost(new_position, hazards) > 0.1 or np.random.random() < 0.3):
                stored += 1
                mpc_learner.store_transition(robot_state, action, flat_hazard_vector, new_robot_state)
            total += 1
            if save and np.random.random() < 0.3:
                mpc_learner.train_models()
            env_state = new_env_state
            robot_state = new_robot_state

            if render:
                env.render()

        print(f"Episode {i}, reward {episode_reward}, cost {episode_cost}, stored {stored}/{total}")

        if save and i != 0 and not i%(EPISODES//5):
            mpc_learner.save("mpcmodel")

    if save:
        mpc_learner.save("mpcmodel")
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
    #loaded_learner = pickle.load(open("mpcmodel", "rb"))
    main(EPISODES=200, mpc_learner=None, render=False, policy=human_policy, save=True)