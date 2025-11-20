# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

###############################################################################

import numpy as np

from time import perf_counter # perf_counter_ns

###############################################################################

from acas.acasxu_basics import get_act_idx, ACT_NAMES, COC_IDX, SR_IDX, SL_IDX, HorizontalObservation


###############################################################################

# 0: rho, distance (ft) [0, 60760]
# 1: theta, angle to intruder relative to ownship heading (rad) [-pi,+pi]
# 2: psi, heading of intruder relative to ownship heading (rad) [-pi,+pi]
# 3: v_own, speed of ownship (ft/sec) [100, 1145? 1200?] 
# 4: v_int, speed in intruder (ft/sec) [0? 60?, 1145? 1200?] 


###############################################################################

class AbstractAgent():

    def __init__(self, name:str=None):
        self.name = name
        self.last_decision_inference_time_s = 0.0
        self.cumulated_decision_inference_time_s = 0.0
        self.max_decision_inference_time_s = 0.0

    def get_name(self):
        if self.name is not None:
            return self.name
        else:
            return "Agent"

    def reset(self, obs):
        pass
        
    def step(self, obs):
        pass
            
    def __str__(self):
        return self.get_name()

#################################################


class RandomAgent(AbstractAgent):

    def __init__(self, default_seed=None, name:str="Random"):
        super().__init__(name)
        self.default_seed = default_seed
        self.rndgen = np.random.default_rng(default_seed)
        
    def reset(self, obs):
        return self.rndgen.choice(range(5))
        
    def step(self, obs):
        return self.rndgen.choice(range(5))

#################################################

class ConstantAgent(AbstractAgent):

    def __init__(self, action=COC_IDX, name:str=None):

        super().__init__(name)

        self.action = get_act_idx(action)
        if self.action is None:
            self.action = COC_IDX

        if self.name is None:
            self.name = "Constant-" + ACT_NAMES[self.action]

    def reset(self, obs):
        return self.action
        
    def step(self, obs):
        return self.action
     

#################################################
    
class ListAgent(AbstractAgent):
    
    def __init__(self, actions=[COC_IDX,COC_IDX,COC_IDX,COC_IDX,COC_IDX,SR_IDX,SR_IDX], mode='cycle', name:str="Fixed-Behavior-Agent"):
        super().__init__(name)
        self.actions = actions
        self.mode = mode
        self.t = None
        
    def reset(self, obs):
        self.t = 0
        return self.actions[self.t]   
        
    def step(self, obs):
        self.t += 1
        if self.mode=='cycle':
            return self.actions[self.t % len(self.actions)]
        else:
            return self.actions[max(self.t, len(self.actions)-1)]
        
    
#################################################


class UtilityModelAgent(AbstractAgent):

    def __init__(self, name:str="Utility-Agent", choice_function=np.argmin, register_inference_time=True, model=None):
        super().__init__(name)
        self.model = model
        self.obs = None
        self.register_inference_time = register_inference_time
        self.choice_function = choice_function
        
    def reset(self, obs):
        if self.register_inference_time:
            self.last_decision_inference_time_s = 0
            self.cumulated_decision_inference_time_s = 0
            self.max_decision_inference_time_s = 0
        self.obs = obs
        #self.action = 0     #default initial action COC
        values = self.model.predict(obs)
        self.action = self.choice_function(values)
        return self.action   
        
    
    def step(self, obs):

        self.obs = obs
        #self.obs = [rho, theta, psi, v_own, v_int]
        ### v_own, v_int, theta, psi, rho = self.obs
        #rho, theta, psi, v_own, v_int = self.obs
        
        if self.register_inference_time:
            time_start = perf_counter()
        
        values = self.model.predict(obs)
        self.action = self.choice_function(values)
        
        if self.register_inference_time:
            time_end = perf_counter()
            self.last_decision_inference_time_s = time_end - time_start
            self.cumulated_decision_inference_time_s += self.last_decision_inference_time_s
            self.max_decision_inference_time_s = max(self.last_decision_inference_time_s, self.max_decision_inference_time_s)
        
        return self.action   



#################################################
        
#rho, theta, psi, v_own, v_int = state7_to_state5(self.vec, self.v_own, self.v_int)

# 0: rho, distance
# 1: theta, angle to intruder relative to ownship heading
# 2: psi, heading of intruder relative to ownship heading
# 3: v_own, speed of ownship
# 4: v_int, speed in intruder

# min inputs: 0, -3.1415, -3.1415, 100, 0
# max inputs: 60760, 3.1415, 3,1415, 1200, 1200

#'Valid range" [100, 1145]'
#v_own = 800 # ft/sec

#'Valid range: [60, 1145]'
#v_int = 500

###############################################################################

#def consolidate_agents(agents):
#
#   consolidated_agents = []
#   
#   for i, a in enumerate(agents):
#      if get_act_idx(a) is not None:
#         consolidated_agents.append(ConstantAgent(action=a))
#      elif a in ['dubins', 'DUBINS', 'Dubins', 'onnx', 'ONNX']:
#         consolidated_agents.append(DubinsAgent())
#      elif a in ['aidge', 'AIDGE', 'Aidge']:
#        consolidated_agents.append(AidgeAgent())
#     elif a in ['lut', 'LUT']:
#        consolidated_agents.append(LutAgent())
#     elif a in ['random', 'rnd', 'rand', 'RANDOM', 'RAND', 'RND']:
#        consolidated_agents.append(RandomAgent())
#     else:
#        consolidated_agents.append(a)
#        
#  return consolidated_agents


###############################################################################


if __name__ == "__main__":

    def run(ag):
        print("---")
        a = ag.reset(obs)    
        print(ACT_NAMES[a])
        for t in range(5):
            a = ag.step(obs)
            print(ACT_NAMES[a])
    
    obs = HorizontalObservation(last_a=COC_IDX, rho=1000, theta=0.0, psi=0.0, v_own=800.0, v_int=700.0)
    print("---")
    print(obs)
    
    agents = [RandomAgent(), ConstantAgent()]
    
    for ag in agents:
        run(ag)
    
    # ---------------------------------------------------
    
    from acasxu_model_lut import NearestLUT
    from acasxu_model_onnx_dubins import DubinsNN

    agents = [UtilityModelAgent(model=DubinsNN()), UtilityModelAgent(model=NearestLUT())]
    
    for ag in agents:
        run(ag)

    