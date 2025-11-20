# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:30:04 2024

@author: fperotto
"""

##################################################################################################

import numpy as np
from acas.acasxu_episode_simulator import AcasSim
from acas.acasxu_basics import Airplane, create_random_intruder
from acas.acasxu_env import HorizontalAcasXuEnv
from acas.acasxu_agents import DubinsAgent, ConstantAgent, LutAgent
from acas.acasxu_gui_pygame import ACASPyGameGUI

n = 20

airplanes = [Airplane(x=0.0, y=0.0, head=0.0)]

rng = np.random.default_rng(7)

for i in range(n-1):
    airplanes.append(create_random_intruder(airplanes[i], incident_time=(40, 90), incident_distance=(500.0, 2000.0), rng_or_seed=rng))
 
max_steps = 120
fps = 20

agents = [ConstantAgent()] * n

env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

sim = AcasSim(env, agents)

gui = ACASPyGameGUI(sim, fps=fps, close_on_finish=False)
gui.launch(start_running=True, window_caption="Multiple Acas-Xu - CONSTANT AGENT")


agents = [DubinsAgent()] * n

sim = AcasSim(env, agents)
sim.reset()

gui = ACASPyGameGUI(sim, fps=fps, close_on_finish=False)
gui.launch(start_running=True, window_caption="Multiple Acas-Xu - ONNX AGENT")



agents = [LutAgent()] * n

sim = AcasSim(env, agents)
sim.reset()

gui = ACASPyGameGUI(sim, fps=fps, close_on_finish=False)
gui.launch(start_running=True, window_caption="Multiple Acas-Xu - LUT AGENT")
