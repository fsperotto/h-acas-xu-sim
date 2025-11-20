# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:49:38 2024

@author: fperotto
"""

###############################################################################

import numpy as np

from gymnasium import Env, spaces

from acas.acasxu_basics import HorizontalAirplane
from acas.acasxu_env import HorizontalAcasXuEnv
from acas.acasxu_agents import ConstantAgent 

###############################################################################

class HorizontalAcasXuGymEnv(HorizontalAcasXuEnv, Env):

    def __init__(self, 
                 #render_mode: str=None,
                 airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=800.0, name='own'), 
                              Airplane(x=10000.0, y=5000.0, head=-3.0, speed=700.0, name='intruder')],
                 agents=None,
                 save_states=False,
                 default_max_steps=200,
                 nmac_distance=500.,      # in ft
                 own_index=0
                 ):
    
        self.own_index = own_index
        
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(
           low=np.array([0, -200000, 0,    0,   -np.pi, -np.pi]),
           high=np.array([4, 200000, 1200, 1200, np.pi,  np.pi]),
           #low=np.array([0, -200000, 0,    0,   -np.pi, -np.pi, 0]),
           #high=np.array([4, 200000, 1200, 1200, np.pi,  np.pi, 9]),
           dtype=np.float32
        )
        
        if agents is None:
            self.agents = [ConstantAgent() for _ in range(len(airplanes))]
        else:
            self.agents = agents
        
        HorizontalAcasXuEnv.__init__(self, airplanes=airplanes, save_states=save_states, default_max_steps=default_max_steps, nmac_distance=nmac_distance)
            

    #--------------------------------------------------------------------------        

    def reset(self, *, seed:int=None, max_steps:int=None, reset_strategy="smart") -> tuple:
    #def reset(self):
        Env.reset(self, seed=seed)
        self.joint_obs = HorizontalAcasXuEnv.reset(self, seed=seed, max_steps=max_steps)
        self.joint_action = [agent.reset(self.joint_obs[i]) for i, agent in enumerate(self.agents)]
        return self.joint_obs[self.own_index], self.get_info()


    #--------------------------------------------------------------------------        

    def step(self, own_action) -> tuple:
        if self.ready and not self.done:
            self.joint_action[self.own_index] = own_action
            self.joint_obs = HorizontalAcasXuEnv.step(self, self.joint_action)
            if not self.done:
                self.joint_action = [agent.step(self.joint_obs[i]) for i, agent in enumerate(self.agents)]
        else:
            self.joint_obs = self.reset()

        return self.joint_obs[self.own_index], self.rewards[self.own_index], self.terminated, self.truncated, self.get_info()

    #--------------------------------------------------------------------------        

    def get_info(self):
        return {}
    
###############################################################################



if __name__ == '__main__' :

   max_steps = 100
   
   airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0), 
                Airplane(x=50000.0, y=50000.0, head=-np.pi/3, speed=780.0)]
   
   env = HorizontalAcasXuGymEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)
   
   obs, info = env.reset()
   for i in range(max_steps):
       if env.done:
           break
       obs, rewards, term, trunc, info = env.step(0)
       

   print('ok')