# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:30:04 2024

@author: fperotto
"""

##################################################################################################

from argparse import ArgumentParser

##################################################################################################

from acas.acasxu_episode_simulator import run_single_sim
from acas.acasxu_basics import Airplane
from acas.acasxu_agents import consolidate_agents


##################################################################################################

def main():
    'main entry point'

    # parse arguments
    parser = ArgumentParser(description='Run visual grid simulation')
    
    #parser.add_argument("-a", "--airplanes", nargs='+', type=float, default=[[0.0, 0.0, 0.0, 1080.0], [25000.0, 50000.0, -1., 1000.0]], help="list of tuples like [(x, y, theta, v), (...)]")
    parser.add_argument("-x", "--coords-x", nargs='+', type=float, default=[0.0, 25000.0], help="x coordinate of each airplane")
    parser.add_argument("-y", "--coords-y", nargs='+', type=float, default=[0.0, 50000.0], help="y coordinate of each airplane")
    parser.add_argument("-v", "--velocities", nargs='+', type=float, default=[1080.0, 1000.0], help="velocity coordinate of each airplane")
    parser.add_argument("-b", "--bearings", nargs='+', type=float, default=[0.0, -1.0], help="heading angle coordinate of each airplane")
    parser.add_argument("-g", "--agents", nargs='+', type=str, default=['dubins','coc'], help="sequence of type of pilot for each airplane, like: dubins lut coc random")
    parser.add_argument("-t", "--horizon", type=int, default=120, help="Max time horizon.")

    args = parser.parse_args()
    
    airplanes = [Airplane(x=x, y=y, head=b, speed=v) for x, y, b, v in zip(args.coords_x, args.coords_y, args.bearings, args.velocities)] 
    
    print(args.agents)
    agents = consolidate_agents(args.agents)

    run_single_sim(total_time=args.horizon, airplanes=airplanes, agents=agents)
    #run_single_sim(**vars(args))
    

##################################################################################################

if __name__ == '__main__':
    main()

