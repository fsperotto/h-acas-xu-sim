# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:30:04 2024

@author: fperotto
"""

import math
import numpy as np
from matplotlib import pyplot as plt

from acas.acasxu_basics import HorizontalEncounter, Airplane
from acas.acasxu_model_lut import NearestLUT
from acas.acasxu_model_onnx_dubins import DubinsNN
from acas.acasxu_model_so_aidge import AidgeNN
from acas.acasxu_utils import plot_state, plot_best_actions, plot_best_actions_cartesian
#from acas.acasxu_env import  AcasEnv #, DubinsAgent


###############################################################################

   
#COMPARE LUT vs ONNX vs AIDGE_FROM_ONNX

def test_compare(psi=0.5, v_own=400, v_int=1000, last_a=0):

   #agents to test
   ag_list = [{"agent_class":DubinsNN , "name":"ONNX"},
              #{"agent_class":AidgeNN , "name":"AIDGE"},
              {"agent_class":NearestLUT , "name":"LUT"}]
    
   fig, ax = plt.subplots(1, len(ag_list), figsize=(8*len(ag_list),8))
    
   for i, agent in enumerate(ag_list):
       #try:
           #create agent from given class
           model = agent["agent_class"]()
            
           plot_best_actions(psi=psi, v_own=v_own, v_int=v_int, last_a=last_a, model=model, fig=fig, ax=ax[i], show=False)
           #plot_best_actions_cartesian(psi=psi, v_own=v_own, v_int=v_int, last_cmd=last_cmd, max_dist=20000, dist_incr=2000, model=model, fig=fig, show=False)
           #plot_best_actions_cartesian(psi=psi, v_own=v_own, v_int=v_int, last_cmd=last_cmd, max_dist=5000, dist_incr=500, model=model, fig=fig, show=False)

           ax[i].set_aspect('equal')
           #fig.show()
           ax[i].set_title(agent["name"])
       #except:
       #    print(f"Error when using {agent['name']} agent.")

   plt.show()



###############################################################################

if __name__ == '__main__' :
   
    from argparse import ArgumentParser
   
    'main entry point'

    # parse arguments
    parser = ArgumentParser(description='Compare LUT, ONNX, and AIDGE models in a given situation')
    
    parser.add_argument("--last-a", type=int, default=3, help="last action (0=COC, 1, 2, 3, 4)")
    parser.add_argument("--psi", type=float, default=0.5, help="intruder relative heading")
    parser.add_argument("--v-own", type=float, default=400.0, help="own speed")
    parser.add_argument("--v-int", type=float, default=1000.0, help="intruder speed")

    args = parser.parse_args()
   
    test_compare(psi=args.psi, v_own=args.v_own, v_int=args.v_int, last_a=args.last_a)
