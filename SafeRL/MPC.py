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
from ModelLearning import NNModel, create_model
from Controller import human_policy
import matplotlib.pyplot as plt

"""
-ve rotation = clockwise torque
i.e. theta defined in the polar sense.
"""

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class MPCLearner:
    def __init__(self):
        self.MEMORY_SIZE = 50000
        self.MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
        self.REPLAY_MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.MINIBATCH_SIZE = 32
        self.state_dims = 10
        self.hazard_dims = 32
        self.action_dims = 2
        self.combined_dims = self.state_dims+self.action_dims
        self.policy_dims = self.state_dims + self.hazard_dims
        l2_penalty = 0
        dropout = 0
        self.dynamics_model = create_model(self.combined_dims, self.state_dims, "linear", l2_penalty=l2_penalty)
        self.policy = create_model(self.policy_dims, self.action_dims, "tanh", l2_penalty=l2_penalty)

    def __call__(self, robot_state, hazard_vector):
        input_vector = np.expand_dims(np.hstack((robot_state, hazard_vector)), axis=1).T
        return np.squeeze(self.policy.predict(input_vector))

    def model_prediction(self, robot_state, action):
        input_vector = np.expand_dims(np.hstack((robot_state, action)), axis=1).T
        return self.dynamics_model.predict(input_vector)

    def store_transition(self, robot_state, action, hazards, new_robot_state):
        stacked_transition = np.hstack((robot_state, action, hazards, new_robot_state))
        self.REPLAY_MEMORY.append(stacked_transition)
        return

    def train_models(self):
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = np.array(random.sample(self.REPLAY_MEMORY, self.MINIBATCH_SIZE))
        policy_inds = np.r_[:self.state_dims, self.combined_dims:self.combined_dims + self.hazard_dims]

        state_action_vector = minibatch[:, :self.combined_dims]
        next_state_vector = minibatch[:, -self.state_dims:]

        self.dynamics_model.fit(x=state_action_vector,
                                y=next_state_vector,
                                batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False)

        self.policy.fit(x=minibatch[:, policy_inds],
                        y=minibatch[:, self.state_dims:self.combined_dims],
                        batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False)
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
    stacked_state = np.hstack(list(modified_state_dict.values()))
    return np.hstack((position, stacked_state))


def save_results(memory_buffer):
    ensure_dir("policy_data")
    pickle.dump(memory_buffer, open(f"policy_data/replay_memory", "wb"))
    return True

def main(EPISODES, mpc_learner=None, render=False, save=False, policy=human_policy):
    if mpc_learner is None:
        mpc_learner = MPCLearner()

    full = False

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
                action = policy(robot_state, env_state["hazards_lidar"])
            new_env_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_cost += info["cost"]
            new_position = env.robot_pos[:2]
            predicted_state = mpc_learner.model_prediction(robot_state, action)
            new_robot_state = form_state(new_env_state, new_position)

            if save:
                stored += 1
                mpc_learner.store_transition(robot_state, action, env_state["hazards_lidar"], new_robot_state)
            total += 1
            if save and np.random.random() < 0.2:
                mpc_learner.train_models()
            env_state = new_env_state
            robot_state = new_robot_state

            if render:
                env.render()

        print(f"Episode {i}, reward {episode_reward}, cost {episode_cost}, stored {stored}/{total}")

        if save and i != 0 and not i%(EPISODES//5):
            save_results(mpc_learner.REPLAY_MEMORY)
            mpc_learner.save("mpcmodel2")

    if save:
        print(len(mpc_learner.REPLAY_MEMORY))
        save_results(mpc_learner.REPLAY_MEMORY)
        mpc_learner.save("mpcmodel2")
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

    training = True
    if training:
        main(EPISODES=1000, mpc_learner=None, render=False, policy=human_policy, save=True)
    else:
        loaded_learner = pickle.load(open("mpcmodel", "rb"))
        main(EPISODES=10, mpc_learner=loaded_learner, render=True, policy=loaded_learner, save=False)