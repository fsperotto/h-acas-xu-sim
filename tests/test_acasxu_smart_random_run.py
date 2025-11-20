# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:30:04 2024

@author: fperotto
"""

##################################################################################################

from argparse import ArgumentParser

from matplotlib import pyplot as plt

import numpy as np

##################################################################################################

from acas.acasxu_basics import AirplanesHorizontalEncounter, create_random_incident  #, create_random_intruder
from acas.acasxu_env import HorizontalAcasXuEnv
from acas.acasxu_agents import consolidate_agents
from acas.acasxu_episode_simulator import AcasSim
from acas.acasxu_renderer_matplotlib import AcasRender

##################################################################################################   

# parse arguments
parser = ArgumentParser(description='smart random ACAS-Xu run')

parser.add_argument("-s", "--seeds", nargs='+', type=int, default=[0, 35, 1000, 7, 76, 89, 99, 345], help="random seed (possible a list)")
parser.add_argument("-a", "--agents", nargs='+', type=str, default=None, help="list of agent models")

args = parser.parse_args()

if args.agents is None:
    agents=[
        ['lut', 'coc'],
        ['dubins', 'coc'],
        #['dubins', 'dubins'],
        #['lut', 'lut'],
        #['aidge', 'coc'],
        #['aidge', 'aidge']
        ]
else:
    agents = args.agents
    #verify that agents is a list of lists
    if len(np.array(agents).shape) == 1: 
        agents = [agents]

seeds = args.seeds
    
###############################################################################

min_d=0
max_d=5000
incident_time=60
total_time=80
   
       
#SMART RANDOM RUN COMPARING
for seed in seeds:

   #if seed is not None:
   #np.random. seed(seed)

   own, intruder = create_random_incident(incident_time=(20,30), rng_or_seed=seed) 
   enc = AirplanesHorizontalEncounter(own=own, intruder=intruder)
   enc.printf()
   print("incident time = ", enc.calculate_incident_time())
   airplanes = [own, intruder]
   env = HorizontalAcasXuEnv( airplanes=airplanes, save_states=True, default_max_steps=total_time)
   #env = AcasEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

   for i, list_agents in enumerate(agents):

       list_agents = consolidate_agents(list_agents)
    
       sim = AcasSim(env, list_agents)
       sim.reset()
       sim.run()
    
       #offline rendering using matplotlib animation
       fig, ax = plt.subplots(figsize=(8,8))
       renderer = AcasRender(env)
       title = "ACAS-Xu: " + ", ".join([agent.name for agent in list_agents])
       renderer.plot(fig=fig, interval=10, title=title)
