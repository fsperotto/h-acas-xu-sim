# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:30:04 2024

@author: fperotto
"""

import math
import time

from matplotlib import pyplot as plt

from acas.acasxu_basics import ACT_NAMES
from acas.acasxu_model_lut import NearestLUT
from acas.acasxu_model_onnx_dubins import DubinsNN
#from acas.acasxu_model_so_aidge import AidgeNN
from acas.acasxu_utils import plot_state


###############################################################################

def test_models_in_situation(obs, plot=True):


   #define situation
   #last_command = 0 #coc
   #rho, theta, psi, v_own, v_int = 10000, -2.5, 0.5, 400, 1000
   #rho, theta, psi, v_own, v_int = 10000, +2.7, +1, 800, 600
   #rho, theta, psi, v_own, v_int = 499, -3.1416, -3.1416, 0., 50.      
   #state = [last_command, v_own, v_int, theta, psi, rho]

   print("")    
   print("###################################")    
   print("")    

   #print situation
   print("INPUT:", obs)
   print("distance (ft)", round(obs.rho))
   print("own speed (ft/s)", round(obs.v_own, 2))
   print("intruder speed (ft/s)", round(obs.v_int, 2))
   print("initial relative angle (degrees)", round(obs.theta * 180 / math.pi, 2))
   print("intruder cap angle (degrees)", round(obs.psi * 180 / math.pi, 2))

   #agents to test
   ag_list = [{"agent_class":DubinsNN , "name":"ONNX"},
              #{"agent_class":AidgeNN , "name":"AIDGE"},
              {"agent_class":NearestLUT , "name":"LUT"}]
    
    
   for agent in ag_list:
       #try:

           print()
           
           #create agent from given class
           model = agent["agent_class"]()
           
           start_time = time.time()
           costs = model.predict(obs)
           elapsed_time = (time.time() - start_time) * 1000
           action = costs.argmin()

           print(agent["name"], costs, action, ACT_NAMES[action], elapsed_time, "ms")
           
       #except:
       #    print(f'Error: {agent["name"]}')

#   print()
#   model = NearestLUT()
#   nearest_mi = model.nearest_multi_index(encounter)
#   nearest_i = model.nearest_index(encounter)
#   nearest_enc = model.nearest_encounter(encounter)
#   start_time = time.time()
#   costs = model.predict(encounter)
#   elapsed_time = (time.time() - start_time) * 1000
#   action = costs.argmin()
#   print("LUT:", costs, action, ACTION_NAMES[action], elapsed_time, "ms")
#   print(" - nearest multi-index=", nearest_mi)
#   print(" - nearest flat index=", nearest_i)
#   print(" - nearest multi-index=", nearest_mi)
#   print(" - nearest values=", nearest_enc)

   print("")    
   print("###################################")    
   print("")    
   
   #plot situation
   if plot:
       fig, ax = plt.subplots(figsize=(8,8))
       plot_state( obs, fig=fig, show=False)
       plt.show()

   
    #   mlt_idx = np.unravel_index(flt_idx, shape)
    #   input_arr = np.array([discrete_values[i][mlt_idx[i]] for i in range(len(shape))] )
    #   return input_arr

       

###############################################################################

if __name__ == '__main__' :
   
    from acas.acasxu_basics import HorizontalObservation
    
    from argparse import ArgumentParser
   
    'main entry point'

    # parse arguments
    parser = ArgumentParser(description='Test LUT, ONNX, and AIDGE models in a given situation')
    
    parser.add_argument("--last-a", type=int, default=0, help="last action (0=COC, 1, 2, 3, 4)")
    parser.add_argument("--rho", nargs='+', type=float, default=10000, help="intruder distance (ft)")
    parser.add_argument("--theta", nargs='+', type=float, default=-2.5, help="intruder relative angle")
    parser.add_argument("--psi", type=float, default=0.5, help="intruder relative heading")
    parser.add_argument("--v-own", type=float, default=400.0, help="own speed (ft/s)")
    parser.add_argument("--v-int", type=float, default=1000.0, help="intruder speed (ft/s)")

    args = parser.parse_args()
   
    obs = HorizontalObservation(last_a=args.last_a, rho=args.rho, theta=args.theta, psi=args.psi, v_own=args.v_own, v_int=args.v_int)
   
    test_models_in_situation(obs)
   
   
   
   