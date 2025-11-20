# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:41:52 2024

@author: fperotto
"""

#import numpy as np

#import stable_baselines3 as sb3
from stable_baselines3 import PPO
#from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.vec_env import SubprocVecEnv

from acas.acasxu_basics import create_random_incident
from acas.acasxu_env import HorizontalAcasXuEnv
from acas.acasxu_gym_env import HorizontalAcasXuGymEnv
from acas.acasxu_episode_simulator import AcasSim
from acas.acasxu_agents import AbstractModel, UtilityModelAgent, ConstantAgent, LutAgent, DubinsAgent 
from acas.acasxu_renderer_matplotlib import AcasRender


from matplotlib import  pyplot as plt


###############################################################################



class SB3Agent(UtilityModelAgent):
    
    def __init__(self, model, name:str="SB3-Agent"):
        super().__init__(name=name)
        self.model = model
        
    def reset(self, obs):
        self.obs = obs
        self.action = 0
        return self.action   

    def step(self, obs):
        self.obs = obs
        print(obs)
        self.action, next_state = self.model.predict(obs, deterministic=True)
        return self.action   


###############################################################################


print("Create Gym Environment")

max_steps = 100

#airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0), 
#             Airplane(x=50000.0, y=50000.0, head=-np.pi/3, speed=780.0)]

airplanes = create_random_incident(rng_or_seed=7)

env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)
gym_env = HorizontalAcasXuGymEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

print("Simulation without learning")

agents=[ConstantAgent(), ConstantAgent()]
sim = AcasSim(env, agents)
sim.reset()
sim.run()

#offline rendering using matplotlib animation
fig, ax = plt.subplots(figsize=(8,8))
renderer = AcasRender(env)
renderer.plot(fig=fig, interval=10)

print("Gym loop with random actions")

episode_over = False
obs, info = gym_env.reset()
while not episode_over:
    action = gym_env.action_space.sample()  # agent policy that uses the observation and info
    obs, reward, terminated, truncated, info = gym_env.step(action)
    episode_over = terminated or truncated

#offline rendering using matplotlib animation
fig, ax = plt.subplots(figsize=(8,8))
renderer = AcasRender(gym_env)
renderer.plot(fig=fig, interval=10)

print("PPO Learn")

#gym_env = AcasGymEnv(airplanes=airplanes, save_states=False, default_max_steps=max_steps)
gym_env.save_states=False

#observation, info = env.reset()

#vec_env = make_vec_env(env, n_envs=4, vec_env_cls=SubprocVecEnv)
#model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")

model = PPO("MlpPolicy", gym_env, verbose=1) #, device="cpu")
model.learn(total_timesteps=100000)

#model._predict = model.predict
#model.predict = lambda obs : model._predict(obs)[0]
#model.save("acas")

print("Resulting Policy")

gym_env.save_states=True

episode_over = False
obs, info = gym_env.reset()
while not episode_over:
    print(obs)
    action, pred_state = model.predict(obs)  # agent policy that uses the observation and info
    obs, reward, terminated, truncated, info = gym_env.step(action)
    episode_over = terminated or truncated

#offline rendering using matplotlib animation
fig, ax = plt.subplots(figsize=(8,8))
renderer = AcasRender(gym_env)
renderer.plot(fig=fig, interval=10)


