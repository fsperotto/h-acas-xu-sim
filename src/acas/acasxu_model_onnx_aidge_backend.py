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
import math
import numpy as np

from acas.acasxu_basics import HorizontalObservation
from acas.acasxu_agents import AbstractModel
   
import aidge_core
import aidge_backend_cpu
import aidge_onnx

from aidge_utils import freeze_producers


#######################################################################


def _load_onnx(last_a=0, tau=0, 
               onnx_folder=None,
               filename_template = None,
               filename_start_index = None,
               backend="cpu", aidge_dtype=None, input_tensor_dimensions=[5],
               apply_recipes=True, freeze=True):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''
    
    if onnx_folder is None:
        #onnx_folder = "../nn/stanford/"        
        onnx_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn/stanford')
        #onnx_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn')

    if filename_template is None:
        filename_template = "ACASXU_run2a_{0}_{1}_batch_2000_dubins.onnx"
    
    if filename_start_index is None:
        filename_start_index = 1
    
    onnx_filename = os.path.join(onnx_folder, filename_template.format(last_a+filename_start_index, tau+filename_start_index))
    #onnx_filename = os.path.join(onnx_folder, "acasxu.nnet.onnx")

    means_for_scaling = [19791.091, 0.0,           0.0,            650.0,   600.0]  #,  7.5188840201005975]
    range_for_scaling = [60261.0,   6.28318530718, 6.28318530718,  1100.0,  1200.0]

    # LOAD ONNX MODEL USING AIDGE
    model = aidge_onnx.load_onnx(onnx_filename)

    if apply_recipes:
        print("APPLYING RECIPES")
        #aidge_core.remove_flatten(model)
        aidge_core.fuse_mul_add(model)
        aidge_core.remove_flatten(model)

    # Freeze the model by setting constant to parameters producers
    if freeze:
        freeze_producers(model)

    if aidge_dtype is None:
        aidge_dtype = aidge_core.dtype.float32

    #set datatype and backend
    model.compile(backend, aidge_dtype, dims=[input_tensor_dimensions])

    # create scheduler
    scheduler = aidge_core.SequentialScheduler(model)
    scheduler.generate_scheduling()
    
    output_nodes = list(model.get_output_nodes())

    return model, scheduler, range_for_scaling, means_for_scaling, output_nodes

    
#######################################################################

DEFAULT_ONNX = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn/stanford/ACASXU_run2a_1_1_batch_2000_dubins.onnx')
DEFAULT_MEANS = np.array([19791.091, 0.0,           0.0,            650.0,   600.0])   #,  7.5188840201005975]
DEFAULT_RANGES = np.array([60261.0,   6.28318530718, 6.28318530718,  1100.0,  1200.0])

class AidgeBackendModel(AbstractModel):

    def __init__(self,
                 filepath=None,
                 max_distance=np.inf,   # max_distance = 68000 ft
                 invert_L_R=True,
                 means_for_centering=None,
                 ranges_for_scaling=None,
                 nptype=np.float32,
                 input_tensor_dimensions=[5],
                 backend='cpu',
                 apply_recipes=True,
                 aidge_dtype='float32',
                 freeze=True):    
    
        super().__init__(max_distance=max_distance, 
                         invert_L_R=invert_L_R,
                         means_for_centering=means_for_centering,
                         ranges_for_scaling=ranges_for_scaling,
                         nptype=nptype)
        
        if aidge_dtype is None or aidge_dtype in [float, 'float32']:
            self.aidge_dtype = aidge_core.dtype.float32
        else:
            self.aidge_dtype = aidge_dtype
        
        self.backend = backend
        
        if filepath is None:
            self.filepath = DEFAULT_ONNX
            self.means_for_centering = DEFAULT_MEANS
            self.ranges_for_scaling = DEFAULT_RANGES
        else:
            self.filepath = filepath

        self._load()


    def _load(self, 
                 input_tensor_dimensions=[5],
                 apply_recipes=True,
                 freeze=True):
        '''load the one neural network'''

        # LOAD ONNX MODEL USING AIDGE
        self.model = aidge_onnx.load_onnx(self.filepath)

        if apply_recipes:
            print("APPLYING RECIPES")
            #aidge_core.remove_flatten(self.model)
            aidge_core.fuse_mul_add(self.model)
            #aidge_core.matmul_to_fc(self.model)
            aidge_core.remove_flatten(self.model)

        # Freeze the model by setting constant to parameters producers
        if freeze:
            freeze_producers(self.model)

        #set datatype and backend
        self.model.compile(self.backend, self.aidge_dtype, dims=[input_tensor_dimensions])

        # create scheduler
        self.scheduler = aidge_core.SequentialScheduler(self.model)
        self.scheduler.generate_scheduling()
        
        output_nodes = list(self.model.get_output_nodes())

        self.output_node = output_nodes[0]


    # def predict(self, obs, ...):
        # SEE PARENT
        # return self._forward(input_data)


    def _forward(self, input_data):
        ''' function called by method predict. While predic prepares input and output, _forward calls model.forward'''

        input_tensor = aidge_core.Tensor(input_data)
        input_tensor.set_datatype(self.aidge_dtype)

        self.scheduler.forward(data=[input_tensor]) 
        
        if self.backend == 'cuda':   #cannot read if backend cuda: segmentation fault
            self.output_node.get_operator().get_output(0).set_backend('cpu')
        
        values = np.array(self.output_node.get_operator().get_output(0))[0]   # why I need to take first element in a list ?
        #action = np.argmin(values)
        
        # make sure to set the  output back to "cuda" otherwise the model will not be usable 
        if self.backend == 'cuda':   #cannot read if backend cuda: segmentation fault
            self.output_node.get_operator().get_output(0).set_backend('cuda')

        if self.nptype is not None:
            values = values.astype(self.nptype)
        
        return values

#######################################################################


class AidgeBackendMultiModel(AbstractModel):

    def __init__(self,
                 onnx_folder=None,
                 filename_template=None,
                 filename_start_index=None,
                 input_tensor_dimensions=[5],
                 apply_recipes=True,
                 backend='cpu',
                 aidge_dtype='float32',
                 freeze=True,    
                 max_distance=np.inf,   # max_distance = 68000 ft
                 invert_L_R=True):    
    
        super().__init__(max_distance=max_distance)
        
        self.invert_L_R = invert_L_R
        self.backend = backend

        if aidge_dtype is None or aidge_dtype in [float, 'float32']:
            self.aidge_dtype = aidge_core.dtype.float32
        else:
            self.aidge_dtype = aidge_dtype
        
        self.nnets = [_load_onnx(last_a=last_a, tau=0,
                                 onnx_folder=onnx_folder, filename_template=filename_template, filename_start_index=filename_start_index,
                                 backend=self.backend, aidge_dtype=self.aidge_dtype, freeze=freeze, apply_recipes=apply_recipes, input_tensor_dimensions=input_tensor_dimensions)
                            for last_a in range(5)]
      

    # RUN INFERENCE USING BACKEND
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

            #get the neural network
            model, scheduler, range_for_scaling, means_for_scaling, output_nodes = self.nnets[obs.last_a]

            # normalized input
            input_tensor = aidge_core.Tensor( (np.array([obs.rho, obs.theta, obs.psi, obs.v_own, obs.v_int]) - means_for_scaling) / range_for_scaling)
            input_tensor.set_datatype(self.aidge_dtype)
            #
            #obs = enc.get_obs()
            #obs = obs[1:]   # get out last_a
            # normalize input
            #for i in range(5):
            #    obs[i] = (obs[i] - means_for_scaling[i]) / range_for_scaling[i]

            scheduler.forward(data=[input_tensor]) 
            
            output_node = output_nodes[0]
            
            if self.backend == 'cuda':   #cannot read if backend cuda: segmentation fault
                output_node.get_operator().get_output(0).set_backend('cpu')
            values = np.array(output_node.get_operator().get_output(0))[0]   # why I need to take first element in a list ?
            #action = np.argmin(values)
            # make sure to set the  output back to "cuda" otherwise the model will not be usable 
            if self.backend == 'cuda':   #cannot read if backend cuda: segmentation fault
                output_node.get_operator().get_output(0).set_backend('cuda')

            ### ATTENTION - THIS ONNX OUTPUT IS APPARENTLY IN A DIFFERENT ORDER COMPARED TO THE LUT TABLE
            ### IF TRUE, MUST REORDER OUTPUTS SWAPING L AND R ELEMENTS !!!!!!!
            #inverting R and L...
            if self.invert_L_R:
                values = values[[0,2,1,4,3]]
    
               
            return values
            

           

#######################################################################


if __name__ == '__main__' :

      
   print()

   nn = DubinsNN()

   #simple test
   last_a = 0 #coc
   rho, theta, psi, v_own, v_int = 10000, +2.7, +1, 800, 600

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
