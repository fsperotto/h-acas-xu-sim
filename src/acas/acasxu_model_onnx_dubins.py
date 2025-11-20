# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:31:09 2024

@author: fperotto

ACASXu neural networks closed loop simulation with Dubins dynamics

Used for falsification, where the opponent is allowed to maneuver over time


-------------------------------------------------------------------------------

The "dubins" ONNX files have been copied from:
    
https://github.com/stanleybak/acasxu_closed_loop_sim

*Stanley Bak:

- "I believe I got them from:"
    
https://github.com/verivital/nnv

https://github.com/trhoangdung

*trhoangdung:

- "Patrick is the one who passed the ACASXu networks from Stanford people"

https://pmusau17.github.io/

*Patrick Musau: 

- "I believe we converted the ACASXu networks to onnx using the Keras to ONNX converter. 
  I wrote a parser from the Stanford format to Keras, and then from there they were parsed into Onnx."

Diego Manzanas (Vanderbilt University) finally explained that:
    
"the source of those ONNX networks are: 
    
https://github.com/dieman95/AcasXu/blob/master/other/functions/ToONNX.m, 

"I do not remember exactly what the problem was, but it had to do with the export version (Opset version in MATLAB), so we had to change the way we did it."

"We used the neural network parser https://github.com/verivital/nnvmt to convert the original files (Stanford format .nnet) to the NNV format 

"this file would convert all original networks to NNV:

https://github.com/dieman95/AcasXu/blob/master/other/functions/AcasxuNNVnet.m

Then, we call this function https://github.com/dieman95/AcasXu/blob/master/other/functions/AcasxuONNX.m to convert all NNV-format networks to ONNX 

this function calls the ToONNX function mentioned in the beginning to convert each file https://github.com/dieman95/AcasXu/blob/master/other/functions/ToONNX.m

You can find all ACAS Xu networks in the different formats here:
 - Stanford .nnet format: https://github.com/dieman95/AcasXu/tree/master/networks/nnet
 - NNV format: https://github.com/dieman95/AcasXu/tree/master/networks/nnv_format
 - ONNX: https://github.com/dieman95/AcasXu/tree/master/networks/onnx

See also:
https://github.com/mldiego/AcasXu

We can find the NNET files in:
https://github.com/guykatzz/ReluplexCav2017/tree/master/nnet

In fact, all the ONNX files in theese repositories are the same:
- https://github.com/dieman95/AcasXu/tree/master/networks/onnx    
- https://github.com/verivital/nnv/tree/master/data/ACASXu/onnx
- https://github.com/stanleybak/acasxu_closed_loop_sim/tree/main/acasxu_dubins



"""

#######################################################################

from functools import partial
import os
#import pathlib 
import math
import numpy as np
import json
import onnx
import onnxruntime as ort

from acas.acasxu_basics import ACT_NAMES, HorizontalObservation
from acas.acasxu_basic_models import AbstractModel
   
#######################################################################

STANFORD_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn/stanford')
STANFORD_ONNX_PATTERN = os.path.join(STANFORD_FOLDER, "ACASXU_run2a_{last_a_idx1}_{tau_idx1}_batch_2000_dubins.onnx")
STANFORD_JSON_PATTERN = os.path.join(STANFORD_FOLDER, "ACASXU_run2a_any_{tau_idx1}_batch_2000_dubins_info.json")

DEFAULT_ONNX = os.path.join(STANFORD_FOLDER, "ACASXU_run2a_1_1_batch_2000_dubins.onnx")
DEFAULT_JSON = os.path.join(STANFORD_FOLDER, "ACASXU_run2a_any_1_batch_2000_dubins_info.json")

STANFORD_INPUT_NAME = 'input'
STANFORD_INPUT_SHAPE = (1, 1, 1, 5)

STANFORD_INPUT_MEANS = np.array([19791.091, 0.0,           0.0,            650.0,   600.0])
STANFORD_INPUT_RANGES = np.array([60261.0,   6.28318530718, 6.28318530718,  1100.0,  1200.0])
STANFORD_OUTPUT_MEAN = 7.5188840201005975
STANFORD_OUTPUT_RANGE = 373.94992

ACETONE_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn/acetone')
#ACETONE_ONNX = os.path.join(ACETONE_FOLDER, 'nn_acas_{last_a_code}_acetone.onnx')
ACETONE_ONNX_PATTERN = os.path.join(ACETONE_FOLDER, 'nn_acas_COC_acetone.onnx')
#ACETONE_JSON = os.path.join(ACETONE_FOLDER, 'nn_acas_{last_a_code}_acetone_info.json')
ACETONE_JSON_PATTERN = os.path.join(ACETONE_FOLDER, 'nn_acas_COC_acetone_info.json')

ACETONE_INPUT_NAME = 'X'
ACETONE_INPUT_SHAPE = (5)

#######################################################################
   
def get_input_dims(onnx_model: onnx.ModelProto) -> list[list]:
    """Return the dimensions of all ONNX inputs.

    :param onnx_model: the onnx model 
    :type onnx_model: onnx.ModelProto
    :return: a list of lists with the shapes of the onnx inputs
    :rtype: list[list]
    """
    #sometimes, the inputs include producers, so they must be excluded
    onnx_producer_nodes_names = [node.name for node in onnx_model.graph.initializer]
    return [[dim.dim_value for dim in node.type.tensor_type.shape.dim] for node in onnx_model.graph.input if not node.name in onnx_producer_nodes_names]

def get_output_dims(onnx_model: onnx.ModelProto) -> list[list]:
    """Return the dimensions of all ONNX outputs.

    :param onnx_model: the onnx model 
    :type onnx_model: onnx.ModelProto
    :return: a list of lists with the shapes of the onnx outputs
    :rtype: list[list]
    """
    return [[dim.dim_value for dim in node.type.tensor_type.shape.dim] for node in onnx_model.graph.output]

def get_input_names(onnx_model: onnx.ModelProto) -> list[str]:
    """Return the names of all ONNX inputs.

    :param onnx_model: the onnx model 
    :type onnx_model: onnx.ModelProto
    :return: a list with the names of the onnx inputs
    :rtype: list[str]
    """
    #sometimes, the inputs include producers, so they must be excluded
    onnx_producer_nodes_names = [node.name for node in onnx_model.graph.initializer]
    return [node.name for node in onnx_model.graph.input if not node.name in onnx_producer_nodes_names]

def get_output_names(onnx_model: onnx.ModelProto) -> list[str]:
    """Return the names of all ONNX outputs.

    :param onnx_model: the onnx model 
    :type onnx_model: onnx.ModelProto
    :return: a list with the names of the onnx outputs
    :rtype: list[str]
    """
    return [node.name for node in onnx_model.graph.output]


#######################################################################

def pattern_to_string(pattern, **kwargs):
    #return pattern.format(last_a_idx=last_a_idx, last_a_idx1=last_a_idx1, last_a_code=last_a_code, tau_idx=tau_idx, tau_idx1=tau_idx1)
    return pattern.format(**kwargs)

#######################################################################

def _load_onnx(#last_a_idx=0, tau_idx=0,
               onnx_filename=DEFAULT_ONNX, 
               json_filename=DEFAULT_JSON, 
               lazy_load=False,
               normalize_inputs=True, 
               denormalize_outputs=True,
               input_shape=None, 
               input_name=None):
    '''load the one neural network as a 3-uple (ort_session, input_ranges_for_scaling, input_means_for_scaling, output_ranges_for_scaling, output_means_for_scaling)'''
      
    #last_a_idx1 = last_a_idx + 1
    #tau_idx1 = tau_idx + 1
    #last_a_code = ACT_NAMES[last_a_idx]

    #onnx_filename = pattern_to_string(onnx_filename_pattern, last_a_idx=last_a_idx, last_a_idx1=last_a_idx+1, last_a_code=ACT_NAMES[last_a_idx], tau_idx=tau_idx, tau_idx1=tau_idx+1)

    input_means_for_centering = None
    input_ranges_for_scaling = None
    output_means_for_centering = None
    output_ranges_for_scaling = None
    
    if json_filename is not None:
        #json_filename = pattern_to_string(json_filename_pattern, last_a_idx=last_a_idx, last_a_idx1=last_a_idx+1, last_a_code=ACT_NAMES[last_a_idx], tau_idx=tau_idx, tau_idx1=tau_idx+1)
        with open(json_filename) as f:
            info = json.load(f)
        if input_name is None:
            if "input_name" in info.keys():
                input_name = info["input_name"]
        if input_shape is None:
            if "input_shape" in info.keys():
                input_shape = info["input_shape"]
        if normalize_inputs:
            if "mean_inputs" in info.keys():
                input_means_for_centering = info["mean_inputs"]
            if "range_inputs" in info.keys():
                input_ranges_for_scaling = info["range_inputs"]
        if denormalize_outputs:
            if "mean_all_outputs" in info.keys():
                output_means_for_centering = info["mean_all_outputs"]
            if "range_all_outputs" in info.keys():
                output_ranges_for_scaling = info["range_all_outputs"]

    if input_shape is None or input_name is None:
        onnx_model = onnx.load(onnx_filename)
        input_shape = get_input_dims(onnx_model)[0]
        input_name = get_input_names(onnx_model)[0]
    
    session = ort.InferenceSession(onnx_filename)

    if not lazy_load: 
        # warm up the network (to avoid lazy loading)
        inputs = {input_name: np.zeros(shape=input_shape, dtype=np.float32)}
        session.run(None, inputs)

    return session, input_ranges_for_scaling, input_means_for_centering, output_ranges_for_scaling, output_means_for_centering

    
def _load_all_last_a_onnx(tau_idx=0,
               onnx_filename_pattern=STANFORD_ONNX_PATTERN, 
               json_filename_pattern=STANFORD_JSON_PATTERN, 
               lazy_load=False,
               normalize_inputs=True, 
               denormalize_outputs=True,
               input_shape=None, 
               input_name=None):
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

#    LOAD NN USING FOR LOOP
#    nets = []
#    for last_cmd in range(5):
#        nets.append(load_network(last_cmd, folder=folder))
#    return nets

#    LOAD NN USING LIST COMPREHENSION
    return [_load_onnx(onnx_filename=pattern_to_string(onnx_filename_pattern, last_a_idx=last_a_idx, last_a_idx1=last_a_idx+1, last_a_code=ACT_NAMES[last_a_idx], tau_idx=tau_idx, tau_idx1=tau_idx+1), 
                       json_filename=pattern_to_string(json_filename_pattern, last_a_idx=last_a_idx, last_a_idx1=last_a_idx+1, last_a_code=ACT_NAMES[last_a_idx], tau_idx=tau_idx, tau_idx1=tau_idx+1), 
                       lazy_load=lazy_load,
                       normalize_inputs=normalize_inputs, 
                       denormalize_outputs=denormalize_outputs,
                       input_shape=input_shape,
                       input_name=input_name) for last_a_idx in range(5)]

#    LOAD NN USING LAZY MAP
#    return map(partial(load_network, folder=folder), range(5))

#    LOAD NN USING MAP (faster method)
#    return list(map(partial(_load_onnx, 
#                            tau_idx=tau_idx, 
#                            onnx_filename=pattern_to_string(onnx_filename_pattern, last_a_idx=last_a_idx, last_a_idx1=last_a_idx+1, last_a_code=ACT_NAMES[last_a_idx], tau_idx=tau_idx, tau_idx1=tau_idx+1), 
#                            json_filename=pattern_to_string(json_filename_pattern, last_a_idx=last_a_idx, last_a_idx1=last_a_idx+1, last_a_code=ACT_NAMES[last_a_idx], tau_idx=tau_idx, tau_idx1=tau_idx+1), 
#                            lazy_load=lazy_load,
#                            normalize_inputs=normalize_inputs, 
#                            denormalize_outputs=denormalize_outputs,
#                            input_shape=input_shape,
#                            input_name=input_name), 
#                    range(5)))


#######################################################################


class DubinsNN(AbstractModel) :

    def __init__(self,
                 onnx_filename=None, 
                 json_filename=None,
                 max_distance=np.inf,   # max_distance = 68000 ft
                 invert_L_R=True,
                 normalize_inputs=True,
                 input_means_for_centering=None,
                 input_ranges_for_scaling=None,
                 denormalize_outputs = True,
                 output_means_for_centering=None,
                 output_ranges_for_scaling=None,
                 nptype=np.float32,
                 input_shape=None,
                 input_name=None):    
    
        super().__init__(max_distance=max_distance, 
                         invert_L_R=invert_L_R,
                         normalize_inputs=normalize_inputs,
                         input_means_for_centering=input_means_for_centering,
                         input_ranges_for_scaling=input_ranges_for_scaling,
                         denormalize_outputs=denormalize_outputs,
                         output_means_for_centering=output_means_for_centering,
                         output_ranges_for_scaling=output_ranges_for_scaling,
                         nptype=nptype)
        
        if onnx_filename is None:
            onnx_filename = DEFAULT_ONNX
            if json_filename is None:
                json_filename = DEFAULT_JSON

        self.onnx_filename = onnx_filename
        self.json_filename = json_filename
        
        if json_filename is not None:
            #json_filename = pattern_to_string(json_filename_pattern, last_a_idx=last_a_idx, last_a_idx1=last_a_idx+1, last_a_code=ACT_NAMES[last_a_idx], tau_idx=tau_idx, tau_idx1=tau_idx+1)
            with open(json_filename) as f:
                info = json.load(f)
            if input_name is None:
                if "input_name" in info.keys():
                    input_name = info["input_name"]
            if input_shape is None:
                if "input_shape" in info.keys():
                    input_shape = info["input_shape"]
            if normalize_inputs:
                if self.input_means_for_centering is None and "mean_inputs" in info.keys():
                    self.input_means_for_centering = info["mean_inputs"]
                if self.input_ranges_for_scaling is None and "range_inputs" in info.keys():
                    self.input_ranges_for_scaling = info["range_inputs"]
            if denormalize_outputs:
                if self.output_means_for_centering is None and "mean_all_outputs" in info.keys():
                    self.output_means_for_centering = info["mean_all_outputs"]
                if self.output_ranges_for_scaling is None and "range_all_outputs" in info.keys():
                    self.output_ranges_for_scaling = info["range_all_outputs"]

        self.normalize_inputs = self.normalize_inputs and self.input_means_for_centering is not None and self.input_ranges_for_scaling is not None
        self.denormalize_outputs = denormalize_outputs and self.output_means_for_centering is not None and self.output_ranges_for_scaling is not None

        if input_shape is None or input_name is None:
            onnx_model = onnx.load(onnx_filename)
            input_shape = get_input_dims(onnx_model)[0]
            input_name = get_input_names(onnx_model)[0]

        self.input_name=input_name
        self.input_shape=input_shape

        self._load()


    def _load(self):
        '''load the one neural network'''
        
        loaded_objects = _load_onnx(onnx_filename=self.onnx_filename, 
                                    json_filename=self.json_filename, 
                                    normalize_inputs=self.normalize_inputs, 
                                    denormalize_outputs=self.denormalize_outputs,
                                    input_shape=self.input_shape,
                                    input_name=self.input_name)
        self.session, self.range_for_input_scaling, self.means_for_input_centering, self.range_for_output_scaling, self.means_for_output_centering = loaded_objects

#        self.session = ort.InferenceSession(self.filepath)
#
#        # warm up the network (to avoid lazy loading)
#        input_data = np.zeros(shape=self.input_shape, dtype=self.nptype)
#        #input_data.shape = self.input_shape
#        self.session.run(None, {self.input_name: input_data})


    # def predict(self, obs, ...):
        # SEE PARENT
        # return self._forward(input_data)


    def _forward(self, input_data):
        ''' function called by method predict. While predic prepares input and output, _forward calls model.forward'''

        input_data = input_data.reshape(self.input_shape)
        #input_data.shape = self.input_shape
        outputs = self.session.run(None, {self.input_name: input_data})
        
        values = outputs[0][0]

        if self.nptype is not None:
            values = values.astype(self.nptype)
        
        return values


#######################################################################

class DubinsNNMultiModel(AbstractModel) :

    def __init__(self, 
                 tau=0, fix_index_for_filename_increment_one=True,
                 onnx_filename_pattern=STANFORD_ONNX_PATTERN, 
                 json_filename_pattern=STANFORD_JSON_PATTERN, 
                 folder=STANFORD_FOLDER, folder_is_relative_to_module=True,
                 lazy_load=False,
                 normalize_inputs=True, denormalize_outputs=True,
                 max_distance=np.inf, 
                 invert_L_R=True):    # max_distance = 68000 ft
    
        super().__init__(max_distance=max_distance,
                         invert_L_R=invert_L_R)

        self.normalize_inputs = normalize_inputs
        self.denormalize_outputs = denormalize_outputs
        
        self.nnets = _load_all_last_a_onnx(tau_idx=tau,
               onnx_filename_pattern=onnx_filename_pattern, 
               json_filename_pattern=json_filename_pattern, 
               lazy_load=lazy_load,
               normalize_inputs=normalize_inputs, 
               denormalize_outputs=denormalize_outputs) 
        
        
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


#######################################################################


if __name__ == '__main__' :

   print('--------------------------')
   print("TEST _load_onnx")
      
   _load_onnx()
   _load_all_last_a_onnx()
   
   _load_onnx(onnx_filename=ACETONE_ONNX_PATTERN, 
              json_filename=ACETONE_JSON_PATTERN,
              normalize_inputs=False, denormalize_outputs=False)
              #input_shapes=[(5)], input_names=['X'])
   
   print('--------------------------')
   print("TEST DubinsNN")

   nn = DubinsNN()

   #simple test
   last_a = 0 #coc
   #rho, theta, psi, v_own, v_int = 10000, +2.7, +1, 800, 600
   rho, theta, psi, v_own, v_int = 10543.0, 2.6704, 0.9425, 834.1, 545.0

   obs = HorizontalObservation(last_a=last_a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)

   print("INPUT:", obs)
   print("distance (ft)", rho)
   print("initial relative angle (degrees)", round(theta * 180 / math.pi, 2))
   print("intruder cap angle (degrees)", round(psi * 180 / math.pi, 2))
   print("own speed (ft/s)", round(v_own, 2))
   print("intruder speed (ft/s)", round(v_int, 2))

   res = nn.predict(obs=obs)
   command = np.argmin(res)

   names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

   print()
   print("OUTPUT (q-values):", res)
   print("OUTPUT DENORMALIZED (q-values):", res * STANFORD_OUTPUT_RANGE + STANFORD_OUTPUT_MEAN)
   print("action:", command, names[command])    

   print('--------------------------')
   print("TEST DubinsNN ACETONE")

   nn = DubinsNN(onnx_filename=ACETONE_ONNX_PATTERN, 
                 json_filename=ACETONE_JSON_PATTERN,
                 normalize_inputs=False, denormalize_outputs=False,
                 invert_L_R=False)

   #simple test
   last_a = 0 #coc
   #rho, theta, psi, v_own, v_int = 10000, +2.7, +1, 800, 600
   rho, theta, psi, v_own, v_int = 10543.0, 2.6704, 0.9425, 834.1, 545.0

   obs = HorizontalObservation(last_a=last_a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)

   print("INPUT:", obs)
   print("distance (ft)", rho)
   print("initial relative angle (degrees)", round(theta * 180 / math.pi, 2))
   print("intruder cap angle (degrees)", round(psi * 180 / math.pi, 2))
   print("own speed (ft/s)", round(v_own, 2))
   print("intruder speed (ft/s)", round(v_int, 2))

   res = nn.predict(obs=obs)
   command = np.argmin(res)

   names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

   print()
   print("OUTPUT (q-values):", res)
   print("action:", command, names[command])    


   print('--------------------------')
   print("EXPORT AS NP TABLE")

   from acasxu_model_lut import get_multi_index_iterator, get_input_values_from_multi_index, INPUT_SHAPE_2D, DISCRETE_VALUES_2D
   from tqdm import tqdm
   from math import prod
   
#    INPUT_SHAPE_2D_SINGLE_LAST_ACT = INPUT_SHAPE_2D[1:]
#    #print(INPUT_SHAPE_2D_SINGLE_LAST_ACT)

#    DISCRETE_VALUES_2D_SINGLE_ACT = DISCRETE_VALUES_2D[1:]
#    #print(DISCRETE_VALUES_2D_SINGLE_ACT)
   
#    OUTPUT_SHAPE_2D_SINGLE_LAST_ACT = INPUT_SHAPE_2D_SINGLE_LAST_ACT + (5,)
#    #print(OUTPUT_SHAPE_2D_SINGLE_LAST_ACT)
   
#    count = prod(INPUT_SHAPE_2D_SINGLE_LAST_ACT)

#    lut_from_onnx = np.zeros(OUTPUT_SHAPE_2D_SINGLE_LAST_ACT, dtype=np.float32)

#    for multi_index in tqdm(get_multi_index_iterator(INPUT_SHAPE_2D_SINGLE_LAST_ACT), total=count):
#        #obs = get_input_values_from_multi_index(multi_index, discrete_values=DISCRETE_VALUES_2D_SINGLE_ACT)
#        #obs = HorizontalObservation([0] + list(obs))
#        #result = nn.predict(obs=obs)
#        ##print(obs, result)
#        #lut_from_onnx[multi_index] = result
#        lut_from_onnx[multi_index] = nn.predict(obs=HorizontalObservation([0] + list(get_input_values_from_multi_index(multi_index, discrete_values=DISCRETE_VALUES_2D_SINGLE_ACT))))
#        #print(index, lut_from_onnx[index])

#    #lut_from_onnx = np.fromfunction(function = lambda idx_rho, idx_theta, idx_psi, idx_v_own, idx_v_int : nn.predict(obs=HorizontalObservation([0] + list(get_input_values_from_multi_index((idx_rho, idx_theta, idx_psi, idx_v_own, idx_v_int), discrete_values=DISCRETE_VALUES_2D_SINGLE_ACT)))), 
#    #                                shape=INPUT_SHAPE_2D_SINGLE_LAST_ACT,
#    #                                dtype=np.float32)    

#    print("done, saving...")
   
#    #np.savez("lut_from_" + onnx_filename, lut_from_onnx=lut_from_onnx) 


   print('--------------------------')
   print("TEST DubinsMultipleNN")

   nn = DubinsNNMultiModel()

   rho, theta, psi, v_own, v_int = 10543.0, 2.6704, 0.9425, 834.1, 545.0

   for a in range(5):

       obs = HorizontalObservation(last_a=a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)
       res = nn.predict(obs=obs)
       command = np.argmin(res)

       print()
       print("OUTPUT (q-values):", res)
       print("OUTPUT DENORMALIZED (q-values):", res * STANFORD_OUTPUT_RANGE + STANFORD_OUTPUT_MEAN)
       print("action:", command, names[command])    

   print('--------------------------')

   # from acasxu_model_lut import DISCRETIZATION, INPUT_SHAPE_2D, INPUT_NAMES_2D, ACT_COC, ACT_SL, ACT_SR, ACT_WL, ACT_WR
   # from itertools import product
   
   # print("Counting...")
   # counter = [0,0,0,0,0]
   # total = 0
   # for multi_index in product( *map(range, INPUT_SHAPE_2D) ):
       # input_values = [DISCRETIZATION[name][multi_index[i]] for i, name in enumerate(INPUT_NAMES_2D)]
       # costs = nn.predict(obs=input_values)
       # act = np.argmin(costs)
       # counter[act] += 1
       # total += 1
       # if total % 100 == 0:
           # print(total, end=' ')

   # print("COC:", counter[ACT_COC], round(counter[ACT_COC]/total*100,2), "%")
   # print("WR:",  counter[ACT_WR],  round(counter[ACT_WR]/total*100,2),  "%")
   # print("WL:",  counter[ACT_WL],  round(counter[ACT_WL]/total*100,2),  "%")
   # print("SR:",  counter[ACT_SR],  round(counter[ACT_SR]/total*100,2),  "%")
   # print("SL:",  counter[ACT_SL],  round(counter[ACT_SL]/total*100,2),  "%")
