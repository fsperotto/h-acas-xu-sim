import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
#from gymnasium.utils.env_checker import check_env
#from gym.utils import seeding

import pygame
import pygame.font

import random
from typing import Optional, Union
#import math

import os
#import time

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg  
#from matplotlib import patches, animation
#from matplotlib.transforms import Affine2D

LENGTH =1200
WIDTH = 800

ACT_COC = 0
ACT_WL = 1
ACT_WR = 2
ACT_SL = 3
ACT_SR = 4

register(
    id='acas-v2',
    entry_point='acas_xu:AcasEnv'
)


class AcasEnv(gym.Env):

    """
    ### Description:
    The AcasEnv class is a custom environment designed to simulate a scenario where two airplanes
    (ownship and intruder) are flying in a 2D plane. The goal is to avoid the collision between 
    the airplanes by adjusting the heading of the ownship based on the relative position, speed, 
    and heading of the intruder.

    The environment is compatible with OpenAI Gym and can be used for reinforcement learning tasks.
    It includes methods for resetting the environment, taking a step in the environment, rendering
    the current state, and closing the environment.

    ### Action Space:
    The action space is a discrete space with 5 possible actions:
    - 0: Maintain current heading
    - 1: Turn left (small angle)
    - 2: Turn right (small angle)
    - 3: Turn left (large angle)
    - 4: Turn right (large angle)

    ### Observation Space:
    The observation space is a continuous space represented by a 6-dimensional vector:
    - Relative distance to the nearest intruder (clipped between -10,000 and 10,000)
    - Ownship speed (clipped between 0 and 300)
    - Intruder speed (clipped between 0 and 300)
    - Relative angle (theta) between ownship and intruder (clipped between -π and π)
    - Relative heading difference (psi) between ownship and intruder (clipped between -π and π)
    - Last action taken by ownship (0 to 4)

    ### Reward Structure:
    - A small positive reward is given for maintaining the current heading.
    - Penalties are applied for conflicting maneuvers.
    - A large negative reward is given if the distance to the intruder drops below a safety threshold (epsilon).

    ### Metadata:
    - render_modes: ["human", "rgb_array"]
    - render_fps: 60

     
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}  # Slow the animation at 15 FPS

    act_to_angle = [0, np.radians(-1.5), np.radians(+1.5), np.radians(-3.), np.radians(+3.)]

    def __init__(self, 
                 save_states=False,
                 render_mode=None,
                 airplanes=None,  
                 epsilon=153,
                 max_time_steps=200, step_ini=0):
        
        self.last_a = 0
        self.epsilon = epsilon
        self.save_states = save_states
        self.states_history = []
        self.commands_history = []
        self.min_dist = 0
        self.max_time_steps = max_time_steps
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.window_size = (LENGTH, WIDTH)  # Size of the Pygame window
        

        self.first_step = True
        self.done = False
        self.info = {}
        self.current_time_step = step_ini

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
           low=np.array([-1e6, 0, 0, -np.pi, -np.pi, 0]),
           high=np.array([1e6, 300, 300, np.pi, np.pi, 4]),
           dtype=np.float32
        )

        if airplanes is None:
            self.airplanes = self.smart_random_init()
        else:
            self.airplanes = airplanes
        
        self.own = self.airplanes[0]
        self.int = self.airplanes[1]
        self.rho = np.sqrt((self.own.x - self.int.x) ** 2 + (self.own.y - self.int.y) ** 2)

    def update_relations(self):
        self.relative_distances = np.array([[np.hypot(own.x-intruder.x, own.y-intruder.y) for intruder in self.airplanes] for own in self.airplanes])
        try:
            #print(self.intruder.x, self.intruder.y)
            num_airplanes = len(self.airplanes)
            self.relative_distances = np.zeros((num_airplanes, num_airplanes))
            self.relative_angles = np.zeros((num_airplanes, num_airplanes))  
            self.relative_heads = np.zeros((num_airplanes, num_airplanes)) 
            #self.relative_distances = np.array([[np.hypot(own.x-intruder.x, own.y-intruder.y) for intruder in self.airplanes] for own in self.airplanes])
            for i, own in enumerate(self.airplanes):
                for j, intruder in enumerate(self.airplanes):
                    if i != j:
                        self.relative_distances[i, j] = np.hypot(own.x - intruder.x, own.y - intruder.y)
                    else:
                        self.relative_distances[i, j] = np.inf 
            #print('Relative distance shape:', self.relative_distances.shape)
            #print('Relative distance :', self.relative_distances)
            self.nearest_intruder_index = self.relative_distances.argmin(axis=-1)
            #self.relative_angles = np.array([[rad_mod(np.arctan2(intruder.y-own.y, intruder.x-own.x)) for intruder in self.airplanes] for own in self.airplanes])
            for i, own in enumerate(self.airplanes):
                for j, intruder in enumerate(self.airplanes):
                    if i != j:
                        self.relative_angles[i, j] = rad_mod(np.arctan2(intruder.y-own.y, intruder.x-own.x))
                    else:
                        self.relative_distances[i, j] = 0
            #self.relative_heads = np.array([[rad_mod(intruder.head-own.head) for intruder in self.airplanes] for own in self.airplanes])
            for i, own in enumerate(self.airplanes):
                for j, intruder in enumerate(self.airplanes):
                    if i != j:
                        self.relative_heads[i, j] = rad_mod(intruder.head-own.head)
                    else:
                        self.relative_heads[i, j] = 0
            self.rho = np.array([self.relative_distances[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
            self.theta = np.array([self.relative_angles[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
            self.psi = np.array([self.relative_heads[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
            self.v_int = np.array([self.airplanes[self.nearest_intruder_index[i]].speed for i in range(len(self.airplanes))])
            self.min_dist = self.rho.min()
        except Exception as e:
            print("Error during update_relations : ", str(e))
            raise

    def _get_obs(self):
        own = self.airplanes[0]
        intruder = self.airplanes[1]

        rho = np.clip(self.rho[0], -1e4, 1e4).item()
        own_speed = np.clip(own.speed, 0, 300).item()
        intruder_speed = np.clip(intruder.speed, 0, 300)
        theta = np.clip(self.theta[0], -np.pi, np.pi)
        psi = np.clip(self.psi[0], -np.pi, np.pi)
        last_a = np.clip(self.last_a, 0, 4).item()

        obs = np.array([rho, own_speed, intruder_speed, theta, psi, last_a], dtype=np.float32)
        return obs

    def _get_info(self):
        own = self.airplanes[0]
        intruder = self.airplanes[1]
        
        info = {
        'own_position': (own.x, own.y),
         'intruder_position': (intruder.x, intruder.y),
         'distance_to_intruder': self.rho[0]
         }

        return info
    
    def smart_random_init(self, total_time=200):
        airplanes = []
        
        #interest_time = random.randrange(80, total_time - 40)
        interest_time = 100

        #head = np.random.uniform(-np.pi, np.pi)
        head = np.pi/2
        
        speed = np.random.uniform(50, 100)
        #speed = 75
        
        x_t, y_t = 0, 0
        x_0 = x_t - (speed * interest_time * np.cos(head))
        y_0 = y_t - (speed * interest_time * np.sin(head))
        airplanes.append(Airplane(x=x_0, y=y_0, head=head, speed=speed, name="own"))

        #head = np.random.uniform(-np.pi, np.pi)
        #head = 2.0
        head = 0

        speed = np.random.uniform(100, 300)
        #speed = 200
        
        x_t, y_t = 0, 0
        x_0 = x_t - (speed * interest_time * np.cos(head))
        y_0 = y_t - (speed * interest_time * np.sin(head))
        airplanes.append(Airplane(x=x_0, y=y_0, head=head, speed=speed, name="int"))

        return airplanes

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.airplanes = self.smart_random_init()
        self.update_relations()
        
        observation = self._get_obs()
        info = {}
        self.current_time_step = 0
        self.first_step = True  # Reinitialize the flag for every new episode
        if self.render_mode == "human":
            self.render()
        
        self.done = False

        return observation, info

    def step(self, action):
        own = self.airplanes[0]
        intruder = self.airplanes[1]

        own.head = rad_mod(own.head + self.act_to_angle[int(action)])
        
        
        own.x += np.cos(own.head) * own.speed   
        own.y += np.sin(own.head) * own.speed 

        intruder.x += np.cos(intruder.head) * intruder.speed 
        intruder.y += np.sin(intruder.head) * intruder.speed 

        #print(np.degrees(intruder.head))
        #print(np.degrees(own.head))

        self.update_relations()
        #print(self.rho)

        self.current_time_step += 1
        
        reward = 0
            
        if action == ACT_COC:
            #reward += 0.0001
            # reward += 0.0005
            reward += 0.0001

        #strengthening action
        elif ((self.last_a == ACT_WL and action == ACT_SL) or (self.last_a == ACT_WR and action == ACT_SR)):
            reward -= 0.009
            #reward -= 0.5
        
        #reversal 
        elif ((self.last_a == ACT_WL or self.last_a == ACT_SL) and (action == ACT_WR or action == ACT_SR)):
            reward -= 0.1
        
        #reversal 
        elif ((self.last_a == ACT_WR or self.last_a == ACT_SR) and (action == ACT_WL or action == ACT_SL)):
            reward -= 0.1

        #crash 
        if self.rho[-1] < self.epsilon:
            reward = -1.0


        #print(reward)
        #print(action)

        terminated = (self.rho[-1] < self.epsilon)
        truncated = (self.current_time_step == self.max_time_steps)
        
        self.last_a = action
        obs = self._get_obs()
        info = self._get_info()
 
        return obs, reward, terminated, truncated, info
    
    def draw_time(self):
        """Displays elapsed time at top left of screen."""
        elapsed_time = self.current_time_step
        time_text = f'Time: {elapsed_time:.2f} s'
        text_surface = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

    def render(self, render_mode="human"):
        
        if self.screen is None:
            pygame.init()
            self.font = pygame.font.Font(None, 30)
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("ACAS XU Simulation")
            self.clock = pygame.time.Clock()
            airplane_image_path = os.path.dirname(os.path.realpath(__file__)) + '/img/airplane.png'
            self.airplane_image = pygame.image.load(airplane_image_path).convert_alpha()
            self.airplane_image = pygame.transform.scale(self.airplane_image, (20,20))

        #if self.first_step:  # Cleaning the screen after each episode
         #   self.screen.fill((255, 255, 255))  # Put a white screen
         #   self.first_step = False

        own = self.airplanes[0]
        intruder = self.airplanes[1]

        #initial_own = (own.x , own.y)
        #initial_intruder = (intruder.x, intruder.y)

        scale = 0.02  # Adjust the scale compared to the window size
        own_pos = (int(own.x * scale + self.window_size[0] / 2), int(own.y * scale + self.window_size[1] / 2))
        intruder_pos = (int(intruder.x * scale + self.window_size[0] / 2), int(intruder.y * scale + self.window_size[1] / 2))

        rotated_img_own=pygame.transform.rotate(self.airplane_image, -np.degrees(own.head)-90)
        rotated_img_int = pygame.transform.rotate(self.airplane_image, -np.degrees(intruder.head)-90)

        rect_own = rotated_img_own.get_rect(center=own_pos)
        rect_int = rotated_img_int.get_rect(center=intruder_pos)

        self.screen.blit(rotated_img_own, rect_own.topleft)
        self.screen.blit(rotated_img_int, rect_int.topleft)
        self.draw_time()
        #self.draw_dashed_line(self.screen, (0, 0, 0), intruder_pos, (LENGTH/2,WIDTH/2), dash_length=15, width=2)

        #pygame.draw.circle(self.screen, (0, 0, 255), own_pos, 10)
        #pygame.draw.circle(self.screen, (255, 0, 0), intruder_pos, 10)

        pygame.display.flip()

        image = pygame.surfarray.array3d(self.screen)
        self.screen.fill((255, 255, 255))
        self.clock.tick(self.metadata["render_fps"])

        return image

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def draw_dashed_line(self, surface, color, start_pos, end_pos, dash_length=10, width=2):
        """Draws a dashed line between start_pos and end_pos."""
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx, dy = x2 - x1, y2 - y1
        distance = (dx**2 + dy**2) ** 0.5
        dash_num = int(distance // dash_length)

        for i in range(dash_num):
            if i % 2 == 0:  # Drawing one segment out of two 
                start = (x1 + dx * i / dash_num, y1 + dy * i / dash_num)
                end = (x1 + dx * (i + 1) / dash_num, y1 + dy * (i + 1) / dash_num)
                pygame.draw.line(surface, color, start, end, width)

# Fonction to normalize the angle
def rad_mod(angle):
    """return angle between -pi and +pi"""
    return ((angle + np.pi) % (np.pi*2)) - np.pi

# Airplane class
class Airplane():

    """
    ### Description:
    This class can contain multiple airplanes and their parameters :
    x : the first component of the airplane's position on the 2D plane
    y : the second component of the airplane's position on the 2D plane
    head : The airplane's heading
    speed : The airplane's speed, which is fixed
    last_a = last action commited by the airplane
    """
    def __init__(self, x=0.0, y=0.0, head=0.0, speed=1.0, name='airplane'):
        self.x = x   
        self.y = y   
        self.head = head  # in rad
        self.speed = speed  # distance per unit of time (on heading direction)
        self.name = name
        self.last_a = 0

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, head: {self.head}, speed: {self.speed}"





if __name__ == "__main__":


    from gymnasium import envs
    from numpy import random

    rndgen = random.default_rng()
    
    #print(envs.registry.all())

    # Create the environment
    env = AcasEnv(render_mode="human")
    #env = gym.make('AcasEnv')

    # Reset the environment
    obs, info = env.reset()
    
    # Define number of steps
    num_steps = 200
    

    for step in range(num_steps):
        # Take action 0 : COC
        action = rndgen.choice([0,1,2,3,4])
        obs, reward, terminated, truncated, info = env.step(action)
    
        # Render the environment for vizalisation
        env.render()
    
        # Check if the environment is terminated or truncated
        if terminated or truncated:
            print(f"Episode terminé à l'étape {step + 1}")
            break
    
    # Close the environment for ressources
    env.close()


