# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

###############################################################################

import numpy as np

from types import SimpleNamespace

###############################################################################

from acas.acasxu_basics import DEFAULT_NMAC_DIST, DEFAULT_MAX_EPISODE_TIME
from acas.acasxu_basics import HorizontalAirplane, HorizontalObservation, HorizontalEncounter, rad_mod
from acas.acasxu_basics import COC_IDX, WL_IDX, WR_IDX, SL_IDX, SR_IDX, ACT_OMEGA
# ORDER OF OBSERVATIONS IN THE ACAS-XU ENVIRONEMENT
from acas.acasxu_basics import OBS_OMEGA_IDX, OBS_RHO_IDX, OBS_THETA_IDX, OBS_PSI_IDX, OBS_V_OWN_IDX, OBS_V_INT_IDX


###############################################################################

# rho, distance (ft) [0, 60760]
# theta, angle to intruder relative to ownship heading (rad) [-pi,+pi]
# psi, heading of intruder relative to ownship heading (rad) [-pi,+pi]
# v_own, speed of ownship (ft/sec) [100, 1145? 1200?] 
# v_int, speed in intruder (ft/sec) [0? 60?, 1145? 1200?] 

###############################################################################
    
class HorizontalAcasXuEnv():

    #--------------------------------------------------------------------------
    def __init__(self, 
                 #render_mode: str=None,
                 #x_int, y_int, head_int, v_int=500,           # in ft, ft, rad, ft/sec
                 #x_own=0., y_own=0., head_own=0., v_own=800,  # in ft, ft, rad, ft/sec
                 airplanes = [HorizontalAirplane(name='own'), HorizontalAirplane(x=10000.0, y=5000.0, heading=-3.0, name='intruder')],
                 save_states=False,
                 default_max_steps=int(DEFAULT_MAX_EPISODE_TIME),
                 nmac_distance=DEFAULT_NMAC_DIST,      # 500ft
                 #decision_freq=1.0,      # in s
                 #update_freq=0.1,        # in s
                 #init_command=0
                 ):      # initial command is COC


        self.current_step = 0
        
        self.ready = False
        self.done = False
        self.terminated = False
        self.truncated = False
        
        self.nmac_distance = nmac_distance

        self.airplanes = airplanes
        self.n = len(self.airplanes)
        
        self.encounters = [[HorizontalEncounter(own=own, intruder=intruder) for own in self.airplanes] for intruder in self.airplanes]

        self.relative_distances = None
        self.nearest_intruder_index = None
        self.rho_nearest =   None
        self.theta_nearest = None
        self.psi_nearest =   None

        # these are set when simulation() if save_states=True
        self.save_states = save_states
        self.states_history = [] # state history
        self.commands_history = [] # commands history

        #joint actions, rewards and observations
        self.previous_actions = None
        self.current_actions = None
        self.rewards = None
        self.gains = None
        self.observations = None

        self.min_dist = float('inf')
        
        self.default_max_steps = default_max_steps
        
        self.renderer = None
        
        #self.reset()


    #--------------------------------------------------------------------------
    def __str__(self):
        return "\n".join([f"x: {airplane.x}, y: {airplane.y}, head: {airplane.head_rad}" for airplane in self.airplanes])


    #--------------------------------------------------------------------------
    def _update_relations(self):

        #could be a triangular matrix representation
        for i in range(len(self.airplanes)):
            for j in range(len(self.airplanes)):
                self.encounters[i][j].update_from_airplanes(self.airplanes[i], self.airplanes[j])
                
        self.relative_distances = np.array([[self.encounters[i][j].rho if i!=j else np.inf for i in range(self.n)] for j in range(self.n)])
        self.nearest_intruder_index = self.relative_distances.argmin(axis=-1)
        
        self.rho_nearest =   [self.encounters[i][self.nearest_intruder_index[i]].rho   for i in range(self.n)]
        self.theta_nearest = [self.encounters[i][self.nearest_intruder_index[i]].theta for i in range(self.n)]
        self.psi_nearest =   [self.encounters[i][self.nearest_intruder_index[i]].psi   for i in range(self.n)]
        #self.relative_angles =    np.array([[rad_mod(np.arctan2(intruder.y-own.y, intruder.x-own.x)) for intruder in self.airplanes] for own in self.airplanes])
        #self.relative_heads =     np.array([[rad_mod(intruder.head-own.head) for intruder in self.airplanes] for own in self.airplanes])
        #self.theta = np.array([self.relative_angles[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
        #self.psi = np.array([self.relative_heads[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
        #self.v_int = np.array([self.airplanes[self.nearest_intruder_index[i]].speed for i in range(len(self.airplanes))])
        
        self.min_dist = self.relative_distances.min()

    #--------------------------------------------------------------------------
    def _update_observations(self):
        for i in range(self.n):
        
            self.observations = [HorizontalObservation([self.previous_actions[i]] + self.encounters[i][self.nearest_intruder_index[i]].to_list()) for i in range(self.n)]

    #--------------------------------------------------------------------------
    def _update_rewards(self):
        for i in range(self.n):
        
            #crash 
            if self.rho_nearest[i] < self.nmac_distance:
                self.rewards[i] = -1.0            
                
            
            elif self.current_actions[i] == COC_IDX:
                #reward += 0.0001
                # reward += 0.0005
                #self.reward[i] = 0.000001
                self.rewards[i] = 0.0001

            #strengthening action
            elif ((self.previous_actions[i] == WL_IDX and self.current_actions[i] == SL_IDX) or (self.previous_actions[i] == WR_IDX and self.current_actions[i] == SR_IDX)):
                self.rewards[i] = -0.009
                #reward -= 0.5
        
            #reversal 
            elif ((self.previous_actions[i] == WL_IDX or self.previous_actions[i] == SL_IDX) and (self.current_actions[i] == WR_IDX or self.current_actions[i] == SR_IDX)):
                self.rewards[i] = -0.1
        
            #reversal 
            elif ((self.previous_actions[i] == WR_IDX or self.previous_actions[i] == SR_IDX) and (self.current_actions[i] == WL_IDX or self.current_actions[i] == SL_IDX)):
                self.rewards[i] = -0.1

            
            self.gains += self.rewards


    #--------------------------------------------------------------------------
    #def reset(self, *, seed:int=None, initial_state=None, options:dict=None) -> tuple:
    #def reset(self):
    def reset(self, *, seed:int=None, max_steps:int=None, reset_strategy="smart"):
        
        self.np_random = np.random.default_rng(seed)
        
        self.current_step = 0
        
        self.previous_actions = [COC_IDX] * self.n
        self.current_actions = [COC_IDX] * self.n
        
        self.max_steps = max_steps
        if max_steps is None: 
            self.max_steps = self.default_max_steps
        
        self.ready = True
        self.done = False
        self.terminated = False
        self.truncated = False
        
        self.rewards = [0.0] * self.n
        self.gains = [0.0] * self.n
        
        for airplane in self.airplanes:
            airplane.reset()
        
        self._update_relations()

        self._update_observations()
        
        self.states_history = [] # state history
        self.commands_history = [] # commands history
        self.dist_history = [] # rho history
        
        if self.renderer is not None:
            self.renderer.refresh()

        return self.observations  #, self._get_info()

    #--------------------------------------------------------------------------
    def step(self, joint_actions=None, reset_if_needed=False):

        if not self.ready or self.done:
            if not reset_if_needed:
                print("[WARNING] should call reset before step.")
            self.reset()
            
        self.current_step += 1
        
        if joint_actions is None:
           joint_actions = [COC_IDX] * self.n
        elif isinstance(joint_actions, int):
           joint_actions = [joint_actions] + [COC_IDX] * (self.n-1)
        
        self.previous_actions = self.current_actions
        self.current_actions = joint_actions
        
        if self.save_states:
            #self.commands.append(self.command)
            #self.int_commands.append(intruder_cmd)
            self.commands_history.append(joint_actions)
            self.states_history.append([[airplane.x, airplane.y, airplane.heading, airplane.v] for airplane in self.airplanes])
            self.dist_history.append(self.min_dist)

        #time_elapse_mat = State.time_elapse_mats[self.command][intruder_cmd] #get_time_elapse_mat(self.command, State.dt, intruder_cmd)

        for i, own in enumerate(self.airplanes):
            own.heading = rad_mod(own.heading + ACT_OMEGA[joint_actions[i]])
            own.x += np.cos(own.heading) * own.v
            own.y += np.sin(own.heading) * own.v

        #update self.encounters...
        self._update_relations()
        
        #update self.rewards...
        self._update_rewards()

        #update self.observations...
        self._update_observations()
                                
        #cur_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2
        #self.rho = np.sqrt((self.x_own - self.x_int)**2 + (self.y_own - self.y_int)**2)
        #own = self.airplanes[0]
        #rho = min([np.sqrt((own.x - intruder.x)**2 + (own.y - intruder.y)**2) for intruder in self.airplanes[1:]])

        #rho = self.relative_distances[self.nearest_intruder_index]
        #theta = self.relative_angles[self.nearest_intruder_index]
        #self.psi = self.relative_heads[self.nearest_intruder_index]
        #self.v_int = self.airplanes[self.nearest_intruder_index].speed
        
        self.terminated = (self.min_dist < self.nmac_distance)
        self.truncated = (self.current_step == self.max_steps)
        self.done = self.terminated or self.truncated
        
        if self.renderer is not None:
            self.renderer.refresh()
         
        #return self.observations, self.rewards, self.terminated, self.truncated  #, self._get_info()
        return self.observations


    #--------------------------------------------------------------------------
    def close(self):
        
        if self.renderer is not None:
            self.renderer.close()
          
    #--------------------------------------------------------------------------
    def get_history(self):
        
        return SimpleNamespace(num_airplanes = self.n,
                               num_steps = len(self.commands_history),
                               commands_history = self.commands_history.copy(), 
                               states_history = self.states_history.copy(), 
                               dist_history = self.dist_history.copy())
            
   
###############################################################################   

if __name__ == '__main__' :
   
   max_steps = 100
   
   airplanes = [HorizontalAirplane(x=0.0, y=0.0, heading=0.0, v=1080.0), 
                HorizontalAirplane(x=50000.0, y=50000.0, heading=-np.pi/3, v=780.0)]
   
   env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

   print()
   print("initial joint observations:")
   joint_observations = env.reset()
   for obs in joint_observations:
       print(obs)
   
   for i in range(5): 
      print()
      print("step", i, " joint observations:")
      joint_observations = env.step()
      for obs in joint_observations:
         print(obs)
      print('joint rewards', env.rewards)
   
   
   
