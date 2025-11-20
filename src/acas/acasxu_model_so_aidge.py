# -*- coding: utf-8 -*-
"""
Created on 19 June 2024

@author: fperotto
"""

###############################################################################

from functools import partial
import os 
import math
from ctypes import cdll, POINTER, c_float
import numpy as np

from acas.acasxu_basics import HorizontalObservation
from acas.acasxu_agents import AbstractModel


###############################################################################

DEFAULT_LIB_SO = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn/libdnn.so')
DEFAULT_MEANS = np.array([19791.091, 0.0,           0.0,            650.0,   600.0])   #,  7.5188840201005975]
DEFAULT_RANGES = np.array([60261.0,   6.28318530718, 6.28318530718,  1100.0,  1200.0])

class AidgeNN(AbstractModel) :

    def __init__(self,
                 filepath=None,
                 max_distance=np.inf,   # max_distance = 68000 ft
                 invert_L_R=True,
                 means_for_centering=None,
                 ranges_for_scaling=None,
                 nptype=np.float32,
                 ctype=c_float):    
    
        super().__init__(max_distance=max_distance, 
                         invert_L_R=invert_L_R,
                         means_for_centering=means_for_centering,
                         ranges_for_scaling=ranges_for_scaling,
                         nptype=nptype)
        
        self.ctype = ctype
        
        if filepath is None:
            self.filepath = DEFAULT_LIB_SO
            self.means_for_centering = DEFAULT_MEANS
            self.ranges_for_scaling = DEFAULT_RANGES
        else:
            self.filepath = filepath

        self._load()


    def _load(self):
        '''load the one neural network'''

        self.model = cdll.LoadLibrary(self.filepath)
        self.model.forward.argtypes = [POINTER(self.ctype), POINTER(self.ctype)]


    # def predict(self, obs, ...):
        # SEE PARENT
        # return self._forward(input_data)


    def _forward(self, input_data):
        ''' function called by method predict. While predic prepares input and output, _forward calls model.forward'''

        input_size = 5

        result_c = (self.ctype * input_size)()
        input_c = (self.ctype * input_size)(*input_data)
    
        #call DNN forward
        self.model.forward(input_c, result_c)

        result_np = np.array(np.fromiter(result_c, dtype=self.nptype, count=input_size))

        return result_np

            
################################################################################################            

def load_network(self, filepath=DEFAULT_LIB_SO):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''

    mylib = cdll.LoadLibrary(filepath)
    mylib.forward.argtypes = [POINTER(c_float), POINTER(c_float)]

    return mylib


def load_networks(folder=None):
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''
#    nets = []
#    for last_cmd in range(5):
#        nets.append(load_network(last_cmd, folder=folder))
#    return nets
    return list(map(partial(load_network, folder=folder), range(5)))  #faster using map 



class AidgeMultiNN(AbstractModel) :

    def __init__(self,
                 lib_folder=None,
                 max_distance=np.inf,   # max_distance = 68000 ft
                 invert_L_R=True):    
    
        super().__init__(max_distance=max_distance)
        
        self.invert_L_R = invert_L_R
    
        self.nnets = load_networks(folder=lib_folder)

        
    def predict(self, 
                obs=None, *,
                last_a=None, rho=None, theta=None, psi=None, v_own=None, v_int=None, tau=None,
                verbose=False):

        obs = HorizontalObservation(obs, last_a=last_a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)
            
        if obs.rho > self.max_distance:

            #print(self.max_distance)
            return np.array([0., +1., +1., +1., +1.])

        else:
        
            if self.invert_L_R:
                obs.last_a = [0, 2, 1, 4, 3][obs.last_a]

            #run the network and return the output
            lib, range_for_scaling, means_for_scaling = self.nnets[obs.last_a]
            
            # normalized input
            input_data = (np.array([obs.rho, obs.theta, obs.psi, obs.v_own, obs.v_int]) - means_for_scaling) / range_for_scaling
        
            #if verbose:
            #    print(f"input (after scaling): {state}")

            #in_array = np.array(state, dtype=np.float32)

            input_size = 5

            result_c = (c_float * input_size)()
            input_c = (c_float * input_size)(*input_data)
        
            #call DNN forward
            lib.forward(input_c, result_c)

            result_np = np.array(np.fromiter(result_c, dtype=np.float32, count=input_size))

            return result_np

###################################################################################################

if __name__ == '__main__' :
   

   from acasxu_basics import HorizontalEncounter


   nn = AidgeNN()

   #simple test
   last_a = 0 #coc
   rho, theta, psi, v_own, v_int = 10000, +2.7, +1, 800, 600

   enc = HorizontalEncounter(last_a=last_a, v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)

   print("INPUT:", enc)
   print("distance (ft)", round(rho, 2))
   print("own speed (ft/s)", round(v_own, 2))
   print("intruder speed (ft/s)", round(v_int, 2))
   print("initial relative angle (degrees)", round(theta * 180 / math.pi, 2))
   print("intruder cap angle (degrees)", round(psi * 180 / math.pi, 2))

   res = nn.predict(enc)
   command = np.argmin(res)

   names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

   print()
   print("OUTPUT (q-values):", res)
   print("action:", command, names[command])    
