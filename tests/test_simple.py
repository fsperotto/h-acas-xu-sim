# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:30:04 2024

@author: fperotto
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from acas.lut_model import NearestLUT, Encounter
from acas.dubins_nn import DubinsNN
from acas.acasxu_sim import plot_state, plot_best_actions, plot_best_actions_cartesian, Airplane, AcasEnv, DubinsAgent


###############################################################################

def test_simple(own_speed=1000.0, intruder_x=100000.0, intruder_y=100000.0, intruder_head=-np.pi/2, intruder_speed=800.0):
    
   total_time = 180
   
   airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=own_speed), 
                Airplane(x=intruder_x, y=intruder_y, head=intruder_head, speed=intruder_speed)]
   
   env  = AcasEnv(airplanes=airplanes, save_states=True)
   
   agents=[DubinsAgent(), DubinsAgent()]
   #agents=[DubinsAgent(), ConstantAgent()]
   #agents=[DubinsAgent(), RandomAgent()]
   #agents=[ConstantAgent(command=0), ConstantAgent(command=1)]

   obs = env.reset()
   actions = [m.reset(obs[i]) for i, m in enumerate(agents)]
   
   print("Airplanes ($t=0$):")
   print("\n".join(str(airplane) for airplane in env.airplanes))
   print("$\\rho$:")
   print(env.relative_distances)
   print(env.rho)
   print("$\\theta$:")
   print(env.relative_angles)
   print("$\\psi$:")
   print(env.relative_heads)

   fig, ax = plt.subplots(figsize=(8,8))

   for i, airplane in enumerate(env.airplanes):
       ax.scatter(airplane.x, airplane.y)   

   for t in range(total_time):
       obs = env.step(actions=actions)
       actions = [m.react(obs[i]) for i, m in enumerate(agents)]
       #print(obs, actions)
   
   #print(env.states_history)
   h = np.array(env.states_history)
   #print(h.shape)
   
   for i, airplane in enumerate(env.airplanes):
       ax.plot(h[:,i,0], h[:,i,1])
   
   ax.set_aspect('equal')
   #fig.show(block=True)
   plt.show(block=True)
   
   print("")    
   print("###################################")    
   print("")    
   

###############################################################################

if __name__ == '__main__' :
   
   test_simple(own_speed=1000.0, intruder_x=100000.0, intruder_y=100000.0, intruder_head=-np.pi/2, intruder_speed=800.0)
