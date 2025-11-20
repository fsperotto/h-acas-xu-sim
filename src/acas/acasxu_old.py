# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

from matplotlib import pyplot as plt

import numpy as np

from acas.acasxu_basics import get_act_idx, Airplane
from acas.acasxu_env import HorizontalAcasXuEnv
from acas.acasxu_episode_simulator import run_single_sim
from acas.acasxu_agents import ConstantAgent, RandomAgent, DubinsAgent, AidgeAgent, LutAgent, consolidate_agents
from acas.acasxu_renderer_matplotlib import AcasRender



       
   
###############################################################################
   
def multiple_smart_random_runs(num_sims=5, seed=None, min_d=0, max_d=5000, interest_time=60, total_time=120, agents=['dubins','coc'], plot=False, save_mp4=False):

    #interesting_seed = -1
    #interesting_state = None

    #num_sims = 10000
    # with 10000 sims, seed 671 has min_dist 4254.5ft

    #start = time.perf_counter()

    agents = consolidate_agents(agents)

    min_d_evolution = [] 
    
    seed_i = None
    for i in range(num_sims):
    #for seed in range(num_sims):
    #    if seed % 1000 == 0:
    #        print(f"{(seed//1000) % 10}", end='', flush=True)
    #    elif seed % 100 == 0:
    #        print(".", end='', flush=True)
    
        if seed is not None:
            seed_i = seed+i

        envs = smart_random_run( seed=seed_i, min_d=min_d, max_d=max_d, interest_time=interest_time, total_time=total_time, agents=agents, plot=plot, save_mp4=save_mp4)
        
        min_d_evolution.append(envs[0].dist_history)
        

        #init_vec, cmd_list, init_velo = make_random_input(seed, intruder_can_turn=intruder_can_turn)
        #
        #v_own = init_velo[0]
        #v_int = init_velo[1]
        #
        ## reject start states where initial command is not clear-of-conflict
        #state5 = state7_to_state5(init_vec, v_own, v_int)
        #
        #if state5[0] > 60760:
        #    command = 0 # rho exceeds network limit
        #else:
        #    res = run_network(State.nets[0], state5)
        #    command = np.argmin(res)
        #
        #if command != 0:
        #    continue
        #
        ## run the simulation
        #s = State(init_vec, v_own, v_int, save_states=False)
        #s.simulate(cmd_list)
        #
        ## save most interesting state based on some criteria
        #if interesting_state is None or s.min_dist < interesting_state.min_dist:
        #    interesting_seed = seed
        #    interesting_state = s

    return min_d_evolution
   
###############################################################################   

if __name__ == '__main__' :
   
   run_single_sim()
   plt.show(block=True)
   
   random_run()
   plt.show(block=True)
   
   smart_random_run()
   plt.show(block=True)

   
   num_sims=30
   seed=3
   min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['dubins', 'coc'], plot=False)
   fig, ax = plt.subplots(figsize=(8,8))
   for d_evo in min_d_evolution:
       plt.plot(d_evo)
   plt.grid()
   plt.show(block=True)

   min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['lut', 'coc'], plot=False)
   fig, ax = plt.subplots(figsize=(8,8))
   for d_evo in min_d_evolution:
       plt.plot(d_evo)
   plt.grid()
   plt.show()

   ###############################   
   

   