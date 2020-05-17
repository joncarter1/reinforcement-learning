import numpy as np
from SafeRL.ModelLearning import create_model, NNModel, early_stopping
import gym
import pickle
from copy import deepcopy
from collections import deque
import random
from keras.utils.vis_utils import plot_model
from SafeRL.Controller import human_policy, safe_policy, config
from safety_gym.envs.engine import Engine
from SafeRL.MPC import form_state, compute_value, compute_cost, preprocess_env_state, preprocess_state2, MPCLearner


class NNPolicy:
    layers = 3
    neurons = 500
    dr = 0.0
    l2_penalty = 1e-4
    lr = 1e-3

    def __init__(self, state_dims, action_dims, existing_model=None):
        self.MEMORY_SIZE = 500000
        self.MIN_REPLAY_MEMORY_SIZE = 3000  # Minimum number of steps in a memory to start training
        self.REPLAY_MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.MINIBATCH_SIZE = 512
        self.state_dims = state_dims
        self.action_dims = action_dims
        if not existing_model:
            self.model = create_model(input_size=self.state_dims,
                                       output_size=self.action_dims,
                                       output_activation="tanh",
                                       probabilistic=False,
                                       neurons=self.neurons,
                                       layers=self.layers,
                                       noise=0,
                                       dr=self.dr,
                                       l2_penalty=self.l2_penalty)
        else:
            self.model = existing_model
        self.input_mean, self.input_std, self.output_mean, self.output_std = [0, 1, 0, 1]
        self.normalised = False

    def __call__(self, state):
        """Neural network policy"""
        normalised_input = (state[np.newaxis, :]-self.input_mean) / self.input_std
        normalised_output = self.model.predict(normalised_input)
        return np.squeeze(normalised_output * self.output_std + self.output_mean)

    def store_transition(self, state, action):
        normalised_state = (state - self.input_mean) / self.input_std
        normalised_action = (action - self.output_mean) / self.output_std
        stacked_transition = np.hstack((normalised_state, normalised_action))
        self.REPLAY_MEMORY.append(stacked_transition)
        return

    def normalise_buffer(self):
        """Normalise all data based on first MIN_REPLAY_MEMORY points for improved NN performance."""
        buffer_array = np.array(self.REPLAY_MEMORY)
        self.input_mean = np.mean(buffer_array[:, :self.state_dims], axis=0)
        self.input_std = np.std(buffer_array[:, :self.state_dims], axis=0)
        self.output_mean = np.mean(buffer_array[:, self.state_dims:], axis=0)
        self.output_std = np.std(buffer_array[:, self.state_dims:], axis=0)
        stacked_mean = np.hstack((self.input_mean, self.output_mean))
        stacked_std = np.hstack((self.input_std, self.output_std))
        buffer_array = (buffer_array - stacked_mean) / stacked_std
        self.REPLAY_MEMORY = list(buffer_array)

    def train_model(self, epochs=5):
        """Train on entire dataset for given number of epochs"""
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        if not self.normalised:
            self.normalise_buffer()
            self.normalised = True

        state_vector = np.array(self.REPLAY_MEMORY)[:, :self.state_dims]
        action_vector = np.array(self.REPLAY_MEMORY)[:, self.state_dims:]

        self.model.fit(x=state_vector, y=action_vector, batch_size=self.MINIBATCH_SIZE,
                        verbose=1, epochs=epochs, validation_split=0.1,
                        shuffle=True, callbacks=[early_stopping])
        return

    def train_on_batch(self, batch_size=None):
        """Train on a small batch of data."""
        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        if not self.normalised:
            self.normalise_buffer()
            self.normalised = True

        if not batch_size:
            batch_size = self.MINIBATCH_SIZE

        minibatch = np.array(random.sample(self.REPLAY_MEMORY, batch_size))
        states = minibatch[:, :self.state_dims]
        actions = minibatch[:, self.state_dims:]
        self.model.fit(states, actions, batch_size=batch_size, verbose=0, shuffle=False)

    def save(self, model_name):
        pickle.dump(self, open(model_name, "wb"))


def main(EPISODES, render=False, save_name=None, policy=human_policy, mpc_model=None):
    nn_policy = NNPolicy(30, 2)

    total_steps = 0
    dangerous_eps = 0
    uncompleted = 0
    for i in range(EPISODES):
        no_accepted = 0
        steps, episode_reward, episode_cost, discount = 0, 0, 0, 1
        gamma = 0.99
        env_state = env.reset()
        goal_pos = env.goal_pos[:2]
        position = env.robot_pos[:2] - goal_pos  # Absolute relative position to goal in x-y terms
        robot_state = form_state(env_state, position)
        if render:
            env.render()
        done = False
        while not done:
            hazards = env.gremlins_obj_pos + env.hazards_pos
            hazard_vector = np.array(hazards)[:, :2]
            policy_state = preprocess_state2(robot_state, hazard_vector)
            if policy == human_policy:
                action = policy(robot_state, goal_pos, hazard_vector).reshape(2,)
                nn_policy.store_transition(policy_state, action)
            else:
                action = policy(policy_state)
            if mpc_model:
                action, found = mpc_model.compute_trajectories(robot_state, goal_pos, hazard_vector, policy)
            new_env_state, env_reward, done, info = env.step(action)
            episode_cost += info["cost"]
            new_position = env.robot_pos[:2] - env.goal_pos[:2]
            new_robot_state = form_state(new_env_state, new_position)
            episode_reward += compute_value(new_robot_state, robot_state)
            discount *= gamma
            robot_state = new_robot_state
            steps += 1
            if render:
                env.render()
        print(f"Episode {i}, reward {episode_reward}, cost {episode_cost}, steps {steps}")
        total_steps += steps
        dangerous_eps += 1 if episode_cost > 0 else 0
        uncompleted += 0 if steps < 1000 else 1
    print(f"Total steps {total_steps}, dangerous eps {dangerous_eps}, uncompleted {uncompleted}")
    if save_name:
        nn_policy.train_model(10)
        nn_policy.save(save_name)
    return


if __name__ == "__main__":
    env = Engine(config)
    nn_policy = pickle.load(open("nn_policy3", "rb"))
    mpc_model = pickle.load(open("MPCModels/mpcmodel", "rb"))
    seed = 1000
    np.random.seed(seed)
    env.seed(seed)
    main(EPISODES=1000, render=True, save_name=None, policy=nn_policy, mpc_model=None) #)
