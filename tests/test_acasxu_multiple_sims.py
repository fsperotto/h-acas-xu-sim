# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:30:04 2024

@author: fperotto
"""

import sys

from matplotlib import pyplot as plt

from acas.acasxu_sim import run_single_sim, random_run, smart_random_run, multiple_smart_random_runs


##################################################################################################

def main():

  try:
    'main entry point'

    #SINGLE RUN
    print("SINGLE RUN")
    run_single_sim()
    
    #COMPLETELY RANDOM RUN
    print("COMPLETELY RANDOM RUN")
    random_run()
    
    #SMART RANDOM RUN
    print("SMART RANDOM RUN")
    smart_random_run(plot_same_figure=False)

    #SMART RANDOM RUN COMPARING
    for seed in [0, 35, 1000, 7, 76, 89, 99, 345]:
       print("COMPARING 4 CASES USING SMART RANDOM RUN WITH SEED", seed)
       smart_random_run(seed=seed, max_d=0, plot_same_figure=False, agents=[['dubins', 'coc'],
                                                                            ['dubins', 'dubins'],
                                                                            #['lut', 'coc'],
                                                                            #['lut', 'lut'],
                                                                            ['aidge', 'coc'],
                                                                            ['aidge', 'aidge']])

    print("MULTIPLE SMART RANDOM SIMULATIONS")

    num_sims=30
   
    min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['dubins', 'coc'], plot=False)
    fig, ax = plt.subplots(figsize=(8,8))
    for d_evo in min_d_evolution:
       plt.plot(d_evo)
    plt.grid()
    plt.show()

    min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['lut', 'coc'], plot=False)
    fig, ax = plt.subplots(figsize=(8,8))
    for d_evo in min_d_evolution:
       plt.plot(d_evo)
    plt.grid()
    plt.show()

    return 0
    
  except:
    plt.close('all')
    sys.exit()

##################################################################################################

if __name__ == '__main__':
    main()

