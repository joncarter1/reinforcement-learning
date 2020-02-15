import safety_gym
import gym
from gym.utils.play import play
import time
import threading
import numpy as np
import pickle
from safety_gym.envs.engine import Engine
from tqdm import tqdm
from keras.models import load_model

state_action_buffer = []
delta_state_buffer = []

model = load_model("model1")


def nn_policy(state):
    stacked_state = np.expand_dims(np.hstack(list(state.values())), axis=1).T

    print(stacked_state.shape)
    return model.predict(x=stacked_state)


def human_policy(new_state):
    """Hand crafted controller for problem."""
    goal_dist = 1 / new_state["goal_dist"]
    goal_compass = new_state["goal_compass"]
    #goal_lidar = new_state["goal_lidar"]
    #print("Distance", goal_dist)
    #print("Compass", goal_compass)
    cos_theta, sin_theta = goal_compass
    forward = cos_theta if goal_dist > 2 else 0.01
    rotation = sin_theta
    return np.array([forward, rotation])


def store_transition(state, action, new_state):
    stacked_state = np.hstack(list(state.values()))
    new_stacked_state = np.hstack(list(new_state.values()))
    state_action_vector = np.hstack((stacked_state, action))

    delta_state_vector = new_stacked_state - stacked_state
    state_action_buffer.append(state_action_vector)
    delta_state_buffer.append(delta_state_vector)

def main(EPISODES, render=False, policy=human_policy):
    #state_action_buffer = []
    #dstate_buffer = []
    for i in tqdm(range(EPISODES)):
        state = env.reset()
        if render:
            env.render()

        done = False
        while not done:
            #action = np.clip(np.random.randn(2), -1, 1)
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            #store_transition(state, action, new_state)
            state = new_state

            if render:
                env.render()

    #print(np.array(state_action_buffer).shape, np.array(delta_state_buffer).shape)
    #pickle.dump(np.array(state_action_buffer), open(f"policy_data/state_action_buffer", "wb"))
    #pickle.dump(np.array(delta_state_buffer), open(f"policy_data/delta_state_buffer", "wb"))


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
        'observe_gremlins': True,
        'constrain_hazards': True,
        'constrain_vases': False,
        'constrain_gremlins': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 6,
        'vases_num': 0,
        'gremlins_num': 4,
        'gremlins_travel': 0.5,
        'gremlins_keepout': 0.4,
    }
    env = Engine(config)
    #env = gym.make('Safexp-PointGoal2-v0')
    main(EPISODES=100, render=True, policy=nn_policy)