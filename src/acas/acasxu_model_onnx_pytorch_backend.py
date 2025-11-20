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

##########################################################
# MODEL ON PYTORCH
##########################################################

#import onnx
import torch
from onnx2torch import convert

print("TORCH CPU", torch.cpu.is_available()) #CPU
print("TORCH XPU (Intel GPU)", torch.xpu.is_available()) #intel GPU
print("TORCH CUDA (Nvidia GPU)", torch.cuda.is_available()) #nvidia GPU

#######################################################################


def _load_onnx(last_a=0, tau=0, 
               onnx_folder=None,
               filename_template = None,
               filename_start_index = None,
               backend="cpu"):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''
    
    if onnx_folder is None:
        #onnx_folder = "../nn/stanford/"        
        #onnx_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn/stanford')
        onnx_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn')

    if filename_template is None:
        filename_template = "ACASXU_run2a_{0}_{1}_batch_2000_dubins.onnx"
    
    if filename_start_index is None:
        filename_start_index = 1
    
    #onnx_filename = os.path.join(onnx_folder, filename_template.format(last_a+filename_start_index, tau+filename_start_index))
    onnx_filename = os.path.join(onnx_folder, "acasxu.nnet.onnx")
    
    #onnx_model = onnx.load(onnx_filename)
    
    means_for_scaling = [19791.091, 0.0,           0.0,            650.0,   600.0]  #,  7.5188840201005975]
    range_for_scaling = [60261.0,   6.28318530718, 6.28318530718,  1100.0,  1200.0]

    # LOAD ONNX MODEL USING ONNX TO PYTORCH
    #model = convert(onnx_model)        
    model = convert(onnx_filename)        
    model.eval()
    model.to(torch.device(backend))

    return model, range_for_scaling, means_for_scaling

    
#######################################################################

class PyTorchModel(AbstractModel):

    def __init__(self,
                 onnx_folder=None,
                 filename_template=None,
                 filename_start_index=None,
                 backend='cpu',
                 max_distance=np.inf,   # max_distance = 68000 ft
                 invert_L_R=True):    
    
        super().__init__(max_distance=max_distance)
        
        self.invert_L_R = invert_L_R
        self.backend = backend
    
        self.nnets = [_load_onnx(last_a=last_a, tau=0,
                                 onnx_folder=onnx_folder, filename_template=filename_template, filename_start_index=filename_start_index, backend=self.backend)
                            for last_a in range(5)]


    # RUN INFERENCE USING PYTORCH
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
            model, range_for_scaling, means_for_scaling, = self.nnets[obs.last_a]



            # normalized input
            input_tensor = torch.from_numpy( (np.array([obs.rho, obs.theta, obs.psi, obs.v_own, obs.v_int]) - means_for_scaling) / range_for_scaling)
            input_tensor = input_tensor.type(torch.FloatTensor)

            #inference
            values = model.forward(input_tensor) 
            #action = np.argmin(values)
                
            ### ATTENTION - THIS ONNX OUTPUT IS APPARENTLY IN A DIFFERENT ORDER COMPARED TO THE LUT TABLE
            ### IF TRUE, MUST REORDER OUTPUTS SWAPING L AND R ELEMENTS !!!!!!!
            #inverting R and L...
            if self.invert_L_R:
                values = values[[0,2,1,4,3]]
    
            return values
            

           

#######################################################################


if __name__ == '__main__' :

      
   print()

   nn = PyTorchModel()

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
