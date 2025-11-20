#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:43:16 2024

@author: fperotto
"""

###################################################################

from time import sleep
import numpy as np 

from matplotlib import pyplot as plt

###################################################################

from acas.acasxu_env import HorizontalAcasXuEnv 
from acas.acasxu_renderer_matplotlib import AcasRender

###################################################################

class AcasSim:
    
    def __init__(self, env, agents, close_when_done=False):
        self.env = env
        self.agents = agents
        self.close_when_done = close_when_done
        self.avg_inference_time = np.zeros(len(agents))
        self.max_inference_time = np.zeros(len(agents))
        
    def reset(self):
        joint_observation = self.env.reset()
        self.joint_action = [agent.reset(joint_observation[i]) for i, agent in enumerate(self.agents)]
        
    def step(self):
        if self.env.ready and not self.env.done:
            joint_observation = self.env.step(self.joint_action)
            if not self.env.done:
                self.joint_action = [agent.step(joint_observation[i]) for i, agent in enumerate(self.agents)]
            else:
                if self.close_when_done:
                    self.env.close()
        else:
            #print("Must reset.")
            self.reset()
        
    def run(self, steps=None, time_delay=None):
        if not self.env.ready or self.env.done:
            #print("Must reset.")
            self.reset()
        t = 0
        while (not self.env.done) and (steps is None or (t < steps)):
            t += 1
            self.step()
            if time_delay is not None:  #delay of step, in seconds, no additional delay if None
                sleep(time_delay)  
        for i, agent in enumerate(self.agents):
            self.avg_inference_time[i] = agent.cumulated_decision_inference_time_s / t
            self.max_inference_time[i] = agent.max_decision_inference_time_s
        print("avg_inference_time", self.avg_inference_time * 1000, "ms")
        print("max_inference_time", self.max_inference_time * 1000, "ms")


###############################################################################
 
def run_single_sim(total_time=80, airplanes=[], agents=[]):
   
   env  = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=total_time)
   
   sim = AcasSim(env, agents)
   sim.reset()
   sim.run()

   renderer = AcasRender(env.get_history())
   renderer.plot()
            
   
###############################################################################   


if __name__ == '__main__' :
   
   from acasxu_basics import Airplane
   from acasxu_agents import ConstantAgent, DubinsAgent, RandomAgent

   print("Run Simulation")
   
   max_steps = 100
   
   airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0), 
                Airplane(x=50000.0, y=50000.0, head=-np.pi/3, speed=780.0)]
   
   agents = [ConstantAgent(), RandomAgent()]
   
   env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

   #env.printf()

   sim = AcasSim(env, agents)
   sim.reset()
   sim.run()

   #obs = env.reset()
   #actions = [agent.reset(obs[i]) for i, agent in enumerate(agents)]
   #
   #for t in range(total_time):
   #    obs, r, tm, tc, info = env.step(actions=actions)
   #    actions = [m.step(obs[i]) for i, m in enumerate(agents)]
   
   print("Gain:", sim.env.gains)

   print()
   print("Run Simulation using run_single_sim")
   airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0), 
                Airplane(x=50000.0, y=50000.0, head=-np.pi/3, speed=780.0)]
   
   run_single_sim(total_time=max_steps, airplanes=airplanes, agents=agents)
   
   
   
