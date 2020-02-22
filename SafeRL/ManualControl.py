import safety_gym
import gym
from gym.utils.play import play
import pygame
import time
import threading
import numpy as np
from safety_gym.envs.engine import Engine

def key_check(l_key, r_key, u_key, d_key):
    presses = pygame.event.get(pump=True)
    if presses != []:
        for event in presses:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_COMMA:
                    l_key = 1
                    l_pressed = 1
                elif event.key == pygame.K_SLASH:
                    r_key = 1
                elif event.key == pygame.K_PERIOD:
                    d_key = 1
                elif event.key == pygame.K_l:
                    u_key = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_COMMA:
                    l_key = 0
                elif event.key == pygame.K_SLASH:
                    r_key = 0
                elif event.key == pygame.K_PERIOD:
                    d_key = 0
                elif event.key == pygame.K_l:
                    u_key = 0

    return l_key, r_key, u_key, d_key


def key_check2():
    l_key, r_key, u_key, d_key = 0, 0, 0, 0
    pressed_keys = np.where(np.array(pygame.key.get_pressed()) == 1)[0]
    if pygame.K_COMMA in pressed_keys:
        l_key = 1
    if pygame.K_SLASH in pressed_keys:
        r_key = 1
    if pygame.K_l in pressed_keys:
        u_key = 1
    if pygame.K_PERIOD in pressed_keys:
        d_key = 1
    return l_key, r_key, u_key, d_key


def main():
    for i in range(10):
        env.reset()
        env.render()
        print(env.observation_space)
        done = False
        i = 0
        l_key, r_key, u_key, d_key = 0, 0, 0, 0

        down = False
        while not done:
            i += 1
            # l_key, r_key, u_key, d_key = key_check(l_key, r_key, u_key, d_key)
            l_key, r_key, u_key, d_key = key_check2()
            pygame.event.pump()
            rotation = 0.5 * (l_key - r_key)
            v_forward = 0.5 * (u_key - d_key)
            # print(v_forward)
            action = (v_forward, rotation)

            new_state, reward, done, lives = env.step(action)
            print(new_state)
            goal_lidar = new_state["goal_lidar"]
            print(goal_lidar)
            env.render()
            clock.tick(50)


if __name__ == "__main__":
    pressed_keys = []
    pygame.init()
    # pygame.display.init()
    clock = pygame.time.Clock()
    env = gym.make('Safexp-PointGoal1-v0')
    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observation_flatten': False,
        'observe_goal_dist': True,
        'observe_goal_comp': True,
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'observe_vases': True,
        'observe_gremlins': True,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 6,
        'vases_num': 3,
        'gremlins_num': 4,
        'gremlins_travel': 0.5,
        'gremlins_keepout': 0.4,
    }
    env = Engine(config)
    # env = gym.make('Safexp-PointGoal2-v0')


    main()

    # play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None)
    # env.unwrapped.viewer.window.on_key_press = key_press
    # env.unwrapped.viewer.window.on_key_release = key_release
    # print(pygame.K_LEFT)
    # print(len(pygame.key.get_pressed()))
