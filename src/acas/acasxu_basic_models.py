# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

In this file are defined the basic classes to create an ACAS-Xu model, 
which receives an ACAS-Xu input (rho, theta, psi, speeds) and returns the output (Q-values)

@author: fperotto
"""

###############################################################################

import numpy as np

#------------------------------------------------------------------------------

from acas.acasxu_basics import HorizontalObservation

###############################################################################

class AbstractModel:
    """
    An ACAS-Xu model is an inference function or engine that returns an output for a given input.
    This class allow to indicate normalization and denormalization parameters (None if no ormalization).
    It is also possible to indicate action reordering (e.g. inverted L and R indexes),
    and also nptype and shape to format input
    """

    def __init__(self, 
                 #obs_dims=['rho', 'theta', 'psi', 'v_own', 'v_int'],    #input for model
                 #sel_dims=['last_a'],                                   #model selector (when multiple)
                 #fix_dims={'tau':0},                                    #fixed input value
                 max_distance = None,     #np.inf,    # max_distance = 68000 ft
                 invert_L_R = False,
                 input_means_for_centering=None,
                 input_ranges_for_scaling=None,
                 output_means_for_centering=None,
                 output_ranges_for_scaling=None,
                 nptype=None, npshape=None):
        """
        Constructor for AbstractModel.
        """
        
        self.max_distance = max_distance

        self.invert_L_R = invert_L_R
        
        self.normalize_inputs = normalize_inputs
        self.input_means_for_centering = input_means_for_centering  
        self.input_ranges_for_scaling  = input_ranges_for_scaling   

        self.denormalize_outputs = denormalize_outputs
        self.output_means_for_centering = output_means_for_centering  
        self.output_ranges_for_scaling  = output_ranges_for_scaling   

        self.nptype = nptype

        self.obs_dims=obs_dims
        self.sel_dims=sel_dims
        self.fix_dims=fix_dims
        
        self.model = None

    #-------------------------------------------------------

    def obs_to_input(self, obs):
        """
        Transform observation into model input data, normalizing and adjusting types.
        """
        
        input_data = np.array([obs.rho, obs.theta, obs.psi, obs.v_own, obs.v_int], dtype=self.nptype)
        
        if self.normalize_inputs:
            if self.input_means_for_centering is not None:
                input_data -= self.input_means_for_centering
            if self.input_ranges_for_scaling is not None:
                input_data /= self.input_ranges_for_scaling
        
        return input_data

    #-------------------------------------------------------

    def output_to_cost(self, output):
        """
        Transform model output into acasxu q-costs, denormalizing and adjusting types.
        """
        
        if self.denormalize_outputs:
            if self.output_ranges_for_scaling is not None:
                output *= self.output_ranges_for_scaling
            if self.output_means_for_centering is not None:
                output += self.output_means_for_centering
        
        if self.nptype is not None:
            output = output.astype(self.nptype)
        
        return output

    #-------------------------------------------------------

    def _forward(self, input_data):
        ''' function called by method predict. While predic prepares input and output, _forward calls model.forward'''
        return None
        
    #-------------------------------------------------------
        
    def predict(self, input_data:np.ndarray):
        """
            input data should be given as a numpy array of floats with 7 elements in this order:
            omega, rho, theta, psi, v_own, v_int, tau
        """
                #obs=None, *,
                #last_a:int=None, 
                #rho:float=None, theta:float=None, psi:float=None, v_own:float=None, v_int:float=None, 
                #tau:float=None,
                #verbose=False):
        
        
        #if obs is None:              
        #    obs = HorizontalObservation(last_a=last_a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)
        #elif type(obs) != HorizontalObservation:
        #    obs = HorizontalObservation(obs)
            
        input_data[RHO_IDX]          
        if obs.rho > self.max_distance:

            return np.array([0., +1., +1., +1., +1.])

        else:
        
            # inverting R and L...
            if self.invert_L_R:
                obs.last_a = [0, 2, 1, 4, 3][obs.last_a]

            # normalized input
            input_data = self.obs_to_input(obs)

            # inference
            output_values = self._forward(input_data)

            # inverting R and L...
            if self.invert_L_R:
                output_values = output_values[[0,2,1,4,3]]
            
            costs = self.output_to_cost(output_values)
            
            return costs


#################################################
    
    
class AbstractMultiModel(AbstractModel):

    def __init__(self,
                 obs_dims=['rho', 'theta', 'psi', 'v_own', 'v_int'],
                 sel_dims=['last_a'],
                 fix_dims={'tau':0},
                 models=None ):

        self.obs_dims=obs_dims
        self.sel_dims=sel_dims
        self.fix_dims=fix_dims
        self.models=models
        
    def predict(self, 
                obs=None, *,
                last_a=None, rho=None, theta=None, psi=None, v_own=None, v_int=None, tau=None,
                verbose=False):        
        pass
        

#################################################

class OnnxModel(AbstractModel):

    def __init__(self, *,
                 onnx_filename="nn/stanford/ACASXU_run2a_1_1_batch_2000_dubins.onnx",
                 json_filename="ACASXU_run2a_any_1_batch_2000_dubins_info.json",
                 max_distance = None,     #np.inf,    # max_distance = 68000 ft
                 invert_L_R : bool = False,
                 normalize_inputs=None,
                 input_means_for_centering=None,
                 input_ranges_for_scaling=None,
                 denormalize_outputs = None,
                 output_means_for_centering=None,
                 output_ranges_for_scaling=None,
                 nptype=None):    

        super().__init__(max_distance=max_distance,
                         invert_L_R=invert_L_R)

#################################################
                 
class DubinsNNMultiModel(AbstractModel) :

    def __init__(self, 
                 tau=0, fix_index_for_filename_increment_one=True,
                 onnx_filename_pattern="ACASXU_run2a_{last_a}_{tau}_batch_2000_dubins.onnx", 
                 json_filename_pattern="ACASXU_run2a_any_{tau}_batch_2000_dubins_info.json", 
                 folder='nn/stanford', folder_is_relative_to_module=True,
                 lazy_load=False,
                 normalize_inputs=True, denormalize_outputs=True,
                 max_distance=np.inf, 
                 invert_L_R=True):    # max_distance = 68000 ft
    
        super().__init__(max_distance=max_distance,
                         invert_L_R=invert_L_R)

        self.normalize_inputs = normalize_inputs
        self.denormalize_outputs = denormalize_outputs
        
        self.nnets = _load_all_last_a_onnx(tau=tau, fix_index_for_filename_increment_one=fix_index_for_filename_increment_one,
               onnx_filename_pattern=onnx_filename_pattern, 
               json_filename_pattern=json_filename_pattern, 
               folder=folder, folder_is_relative_to_module=folder_is_relative_to_module,
               lazy_load=lazy_load,
               normalize_inputs=normalize_inputs, denormalize_outputs=denormalize_outputs) 
        
        
    def predict(self, 
                obs=None, *,
                last_a=None, rho=None, theta=None, psi=None, v_own=None, v_int=None, tau=None,
                verbose=False):

        obs = HorizontalObservation(obs, last_a=last_a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)
            
        if obs.rho > self.max_distance:
    
            return np.array([0, +1, +1, +1, +1])
    
        else:

            if self.invert_L_R:
                obs.last_a = [0, 2, 1, 4, 3][obs.last_a]

            #get the neural network
            session, ranges_for_input_scaling, means_for_input_centering, ranges_for_output_scaling, means_for_output_centering, = self.nnets[obs.last_a]
            
            #obs = enc.get_obs()
            obs = obs[1:]   # get out last_a
            
            # normalize input
            if self.normalize_inputs:
                #obs = (obs - means_for_input_centering) / ranges_for_input_scaling
                for i in range(5):
                    obs[i] = (obs[i] - means_for_input_centering[i]) / ranges_for_input_scaling[i]
        
            if verbose:
                print(f"input (after scaling): {obs}")
    
            in_array = np.array(obs, dtype=np.float32)
            in_array = in_array.reshape((1, 1, 1, 5))
            #in_array.shape = (1, 1, 1, 5)
            outputs = session.run(None, {'input': in_array})
            
            values = outputs[0][0]
            
            ### ATTENTION - THIS ONNX OUTPUT IS APPARENTLY IN A DIFFERENT ORDER COMPARED TO THE LUT TABLE
            ### IF TRUE, MUST REORDER OUTPUTS SWAPING L AND R ELEMENTS !!!!!!!
            #
            #inverting R and L...
            if self.invert_L_R:
                values = values[[0,2,1,4,3]]
    
            return values
            
            
###############################################################################


if __name__ == "__main__":
    
    obs = HorizontalObservation(last_a=COC_IDX, rho=1000, theta=0.0, psi=0.0, v_own=800.0, v_int=700.0)
    print("---")
    print(obs)
    
    model = OnnxModel()

    for t in range(5):
        a = ag.step(obs)
        print(ACT_NAMES[a])