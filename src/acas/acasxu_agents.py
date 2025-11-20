# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

###############################################################################

from acas.acasxu_basic_agents import AbstractAgent, RandomAgent, ConstantAgent, ListAgent, UtilityModelAgent

from acasxu_model_lut import NearestLUT
from acasxu_model_onnx_dubins import DubinsNN

###############################################################################

class DubinsAgent(UtilityModelAgent):
    
    def __init__(self):
        super().__init__(model=DubinsNN())

class LutAgent(UtilityModelAgent):
    
    def __init__(self):
        super().__init__(model=NearestLUT())


###############################################################################

if __name__ == "__main__":

    from acas.acasxu_basics import ACT_NAMES, COC_IDX, HorizontalObservation
        
    obs = HorizontalObservation(last_a=COC_IDX, rho=1000, theta=0.0, psi=0.0, v_own=800.0, v_int=700.0)
    print("---")
    print(obs)
    
    agents = [RandomAgent(), ConstantAgent(), DubinsAgent(), LutAgent()]
    
    for ag in agents:
        print("---")
        a = ag.reset(obs)    
        print(ACT_NAMES[a])
        for t in range(5):
            a = ag.step(obs)
            print(ACT_NAMES[a])
    

    