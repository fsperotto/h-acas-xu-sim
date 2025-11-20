# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:31:09 2024

@author: Filipo S. Perotto (fperotto)
"""

###############################################################################

"""
NOTES ABOUT THESE DATA

From EUROCAE (in fact Mélanie Ducoffe obtained it via Airbus), we obtained a "TableCompressed.npz", having 1.6 Go, 
which is a zipped numpy file containing 5 binary numpy array files (R, L, COC, WR, WL). 

Each uncompressed of those 5 files make 8 Go ! 
It can be difficult to load at once, depending on python configurations.
When numpy cannot read it directly, we need to manipulate the zip file.

The complete information identifying each discrete situation is saved at each row with the q-values, all in float64.
but apparently the q-values can be represented in int16

that file was transmitted by Mélanie Ducoffe, and a script ("orginal_read") to read it by Adrien Gauffriau

in that script, the columns are indicated as: 
    columns=["vertical_tau", "range", "theta", "psi", "ownspeed", "intrspeed",
             "cost_COC", "cost_WR", "cost_WL", "cost_R", "cost_L"]
NOTE: L and R and WL and WR seems to be inverted, in relation to nnet and onnx models ? ***TO VERIFY***

In DeepGreen, we created a function that converts it:
  1) reunifying the 5 separated arrays 
  2) removing the first 6 columns (input)
  3) preserving the last 5 columns (output) but casting to int16

-------------------------------------------------------------------------------

ORDER OF ACTIONS:
    
PHISICALLY, it should be better to represent the order like that -->  SL, WL, COC, WR, SR

IN TERMS OF PREFERENCE IN CASE OF EQUIVALENT COST, AND CONSIDERING CONVENTION FOR TURNING RIGHT  -->  COC, WR, SR, WL, SL   or   COC, WR, WL, SR, SL

In the ONNX files from Stanford, however, the order apparently is COC, WL, WR, SL, SR

Right-of-way rules are described in ICAO Annex 2: Rules of the Air, as follows:
    3.2.1 Right-of-way. The aircraft that has the right-of-way shall maintain its heading and speed.
    3.2.2.1 An aircraft that is obliged by the following rules to keep out of the way of another shall avoid passing over, under or in front of the other, 
            unless it passes well clear and takes into account the effect of aircraft wake turbulence.
    3.2.2.2 Approaching head-on. When two aircraft are approaching head-on or approximately so and there is danger of collision, 
            each shall alter its heading to the RIGHT.            
    3.2.2.3 Converging. When two aircraft (not gliders or balloons) are converging at approximately the same level, 
            the aircraft that has the other on its right shall give way            
    3.2.2.4 Overtaking. An overtaking aircraft is an aircraft that approaches another from the rear on a line 
            forming an angle of less than 70 degrees with the plane of symmetry of the latter.
            An aircraft that is being overtaken has the right-of-way and the overtaking aircraft, whether climbing, descending or in horizontal flight, 
            shall keep out of the way of the other aircraft by altering its heading to the RIGHT.

-------------------------------------------------------------------------------

Claire Pagetti found a zip with (corrupted) .dat files

with the following explanations: 
    
# The purlose of this script is to create a dataset using the ACAS-Xu table that are present in a ZIP archive. 
The documentation of tables is given in Appendix G of the PDF ACAS_ADU_19_001_V5R0a. 
There is only two pages that are not fully coherent with the binary structure. 
Some description in this script are based on reverse engineering of the binary format and may be wrong.

# The Zip archive has been provided by A3. Normally to be able to interact with the table the company has to be part of the Working Group 147 inside the RTCA. 
It shall also be registered as a member of the working group.
Airbus is part of the RTCA and it is very easy to be register in the WG-147.

There are two different zip archives. 
The first one ACAS_PKU_19_002.zip contains table for Vertical Resolution (Xa & Xo) and the parameter file (paramsfile_xu_V5R0_origami_v55ec_h23ec_noMB.txt). 
The second one contains the table for Horizontal conflict resolution (ACAS-Xu). 
The file that contains action cost is q_xuhtrm_v5r0_23ec_allSplits_noMB.dat

Archive:  ACAS_PKU_19_002.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
  2122250  11-20-2018 10:19   entry_vert_xuvtrm_v5r0_compressed.dat
 26128095  11-20-2018 10:19   entry_xuvtrm_v5r0_compressed.dat
      453  03-15-2019 01:11   md5sum_ACAS_PKU_19_002_V5R0.txt
    48860  03-15-2019 00:31   paramsfile_xu_V5R0_origami_v55ec_h23ec_noMB.txt
 70308260  01-31-2019 13:35   q_xuvtrm_v5r0_55ec_allSplits.dat
  2423634  01-31-2019 13:35   q_xuvtrm_v5r0_55ec_minBlocks.dat
  3488071  01-24-2019 10:31   vertical_daa_xuvtrm_v5r0.dat
---------                     -------
104519623                     7 files


Archive:  ACAS_PKU_19_003_V5R0.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
      217  03-15-2019 13:08   md5sum_ACAS_PKU_19_003_V5R0.txt
4342659601  01-31-2019 12:42   q_xuhtrm_v5r0_23ec_allSplits_noMB.dat
472026280  01-31-2019 12:43   q_xuhtrm_v5r0_23ec_minBlocks_noMB.dat
271971216  01-10-2019 15:21   xuhtrm_coord_v5r0.bin
---------                     -------
5086657314                     4 files

# The ACAS-Xu algorithm uses cost table that allow to choose what is the best action to perform depending on input. 
The highest cost gives the action to perform.   ---> HIGHEST ???
The q_xuhtrm_v5r0_23ec_allSplits_noMB.dat contains all the cost table. 
Those are multidimensionnal tables. 
For each point of the grid the table gives the associated cost. 
The architecture of the binary table is the following. 
This is not well documented and based on the reverse of the binary file.

 ==========================
| aux_data |  Block 1
|__________|
|   data   |
|==========|===============
|          |  Block 2
|__________|
|          |
|==========|==============
|          |  Block 3 
|__________|
|          |

...

|==========|
|          | Block 23
|__________|
|          |


The binary files contains 23 blocks. Each block gives costs from a state (COC, WL, WR, L, R) to another state.
For each transition there is a matrix in the parameter file that gives number of block to be used.

|         |COC  |WR  |WL  |R  |L   |
|---------|-----|----|----|---|----|
|**COC**  |1    |6   |6   |15 |15  |                     --->  why R and L have the same costs in this case  ????
|**WR**   |2    |7   |11  |16 |20  |
|**WL**   |3    |8   |12  |17 |21  |
|**R**    |4    |9   |13  |18 |22  |
|**L**    |5    |10  |14  |19 |23  |

CAREFUL : THIS POINT IS NOT WELL DOCUMENTED AND IS MAINLY BASED ON MY UNDERSTANDING
Each block is splitted in 2 parts. The first part is auxiliary data that givies information on how is organised the data part that is the second part. 
The auxiliary data gives the number of dimension, the name of variables, the number of cut for each dimensions and the value for each cut. 
This also gives the number of elements in the data part (that is coherent with the number of cuts in each dimensions!!!)

The data part gives the costs of action for each inputs in the cuts. 
I have assumed that the order of cut is the same as the one given in the auxiliary data part. 
An order is also given in the Appendix G of the PDF document but this is not compatible with the number of data and number of cuts. 
The first value of the data segment should be the scale factor, but this is not relevant with the number of data. 
I have assumed that the scale factor is not present in the data segment. 
The PDF document explains that the type of data is int32, but this not relevant with the size (in bytes) of the data segment. 
The type of data is uint16. This is compatible with the number of bytes and with the text type field that is present in the auxiliary data segment.

The following script perfoms the read of the binary files and create numpy array that may be used for training a network. 
The objective is to have an input vector that contains theta, range, phi, psi, ownspeed, inspeed, lastaction 
and in output that gives the cost for each transition towards a new state. Costs are given for the same input.

# Create table for the transition. This will allow to have the output vector for a cut value
table_COC = np.array([COST_LIST[0],COST_LIST[1],COST_LIST[2],COST_LIST[3],COST_LIST[4]])
table_WR = np.array([COST_LIST[5],COST_LIST[6],COST_LIST[7],COST_LIST[8],COST_LIST[9]])
table_WL = np.array([COST_LIST[5],COST_LIST[10],COST_LIST[11],COST_LIST[12],COST_LIST[13]])
table_R = np.array([COST_LIST[14],COST_LIST[15],COST_LIST[16],COST_LIST[17],COST_LIST[18]])
table_L = np.array([COST_LIST[14],COST_LIST[19],COST_LIST[20],COST_LIST[21],COST_LIST[22]])


# TO VERIFY!  it could have confusion between right and left codes --- must verify it
#
# in the github matlab code of Diego M.L., concerning the NNETs and ONNXs :
#
#function y = advisoryACAS(r)
#    if r == 1                    # 0 COC 
#        y = 0;
#    elseif r == 2                # 1 WL +1.5 positive=clockwise=left
#        y = deg2rad(1.5);
#    elseif r == 3                # 2 WR -1.5 negative=counterclockwise=right
#        y = deg2rad(-1.5);
#    elseif r == 4                # 3 SL +3 positive=clockwise=left
#        y = deg2rad(3.0);
#    elseif r == 5                # 4 SR -3 negative=counterclockwise=right
#        y = deg2rad(-3.0);
#    end
#end
#

# #FROM THE PRECEDENT SOURCE :

# # dtype int16 ?

# #Cut Name : vertical_tau
# #Cut Number : 10
# vertical_tau_arr = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 101.0]
# #Min : 0.000000   Max : 101.000000


# #Cut Name : range (rho)
# #Cut Number : 39
# range_rho_arr = [499.0, 800.0, 2508.0, 4516.0, 6525.0, 8534.0, 10543.0, 12551.0, 14560.0, 16569.0, 18577.0, 20586.0, 22595.0, 24603.0, 26612.0, 28621.0, 30630.0, 32638.0, 34647.0, 36656.0, 38664.0, 40673.0, 42682.0, 44690.0, 46699.0, 48708.0, 50717.0, 52725.0, 54734.0, 56743.0, 58751.0, 60760.0, 75950.0, 94178.0, 112406.0, 130634.0, 148862.0, 167090.0, 185318.0]
# #Min : 499.000000   Max : 185318.000000


# #Cut Name : theta
# #Cut Number : 41
# theta_arr = [-3.1416, -2.9845, -2.8274, -2.6704, -2.5133, -2.3562, -2.1991, -2.042, -1.885, -1.7279, -1.5708, -1.4137, -1.2566, -1.0996, -0.9425, -0.7854, -0.6283, -0.4712, -0.3142, -0.1571, 0.0, 0.1571, 0.3142, 0.4712, 0.6283, 0.7854, 0.9425, 1.0996, 1.2566, 1.4137, 1.5708, 1.7279, 1.885, 2.042, 2.1991, 2.3562, 2.5133, 2.6704, 2.8274, 2.9845, 3.1416]
# #Min : -3.141600   Max : 3.141600


# #Cut Name : psi
# #Cut Number : 41
# psi_arr = [-3.1416, -2.9845, -2.8274, -2.6704, -2.5133, -2.3562, -2.1991, -2.042, -1.885, -1.7279, -1.5708, -1.4137, -1.2566, -1.0996, -0.9425, -0.7854, -0.6283, -0.4712, -0.3142, -0.1571, 0.0, 0.1571, 0.3142, 0.4712, 0.6283, 0.7854, 0.9425, 1.0996, 1.2566, 1.4137, 1.5708, 1.7279, 1.885, 2.042, 2.1991, 2.3562, 2.5133, 2.6704, 2.8274, 2.9845, 3.1416]
# #Min : -3.141600   Max : 3.141600


# #Cut Name : ownspeed
# #Cut Number : 12
# own_spped_arr = [50.0, 87.6, 139.9, 211.0, 307.2, 438.9, 570.6, 702.4, 834.1, 965.8, 1097.5, 1200.0]
# #Min : 50.000000   Max : 1200.000000


# #Cut Name : intrspeed
# #Cut Number : 12
# intr_speed_arr = [0.0, 100.0, 142.0, 203.0, 290.0, 414.0, 545.0, 676.0, 807.0, 938.0, 1069.0, 1200.0]
# #Min : 0.000000   Max : 1200.000000
"""


###############################################################################

import os

import math
import numpy as np

from itertools import product

###############################################################################

from acas.acasxu_basics import COC, WR, WL, SR, SL, ACT_NAMES, OMEGA_SL, OMEGA_WL, OMEGA_COC, OMEGA_WR, OMEGA_SR
from acas.acasxu_basics import OMEGA, RHO, THETA, PSI, V_OWN, V_INT   #, INPUT_NAMES   # TAU,

from acas.acasxu_basics import HorizontalObservation
from acas.acasxu_basic_models import AbstractModel


###############################################################################

#This ACAS-Xu package defines the actions in the following order: SL, WL, COC, WR, SR
#however in the LUT table, as recuperated from EUROCAE, the order is COC, WR, WL, SR, SL
LUT_ACT_ORDER = [COC, WR, WL, SR, SL]

LUT_COC_IDX = LUT_ACT_ORDER.index(COC)  # 0 # clear of conflict, no action
LUT_WR_IDX  = LUT_ACT_ORDER.index(WR)   # 1 # weak right
LUT_WL_IDX  = LUT_ACT_ORDER.index(WL)   # 2 # weak left
LUT_SR_IDX  = LUT_ACT_ORDER.index(SR)   # 3 # strong right
LUT_SL_IDX  = LUT_ACT_ORDER.index(SL)   # 4 # strong left

#the order of actions is different between simulator and LUT
LUT_TO_ACT_IDX = [LUT_ACT_ORDER.index(i) for i in ACT_NAMES]

###############################################################################

#LUT_INPUT_ORDER = [LAST_A, TAU, RHO, THETA, PSI, V_OWN, V_INT]
#
#LUT_LAST_A_IDX = LUT_INPUT_ORDER.index(LAST_A)
#LUT_TAU_IDX =    LUT_INPUT_ORDER.index(TAU)
#LUT_RHO_IDX =    LUT_INPUT_ORDER.index(RHO)
#LUT_THETA_IDX =  LUT_INPUT_ORDER.index(THETA)
#LUT_PSI_IDX =    LUT_INPUT_ORDER.index(PSI)
#LUT_V_OWN_IDX =  LUT_INPUT_ORDER.index(V_OWN)
#LUT_V_INT_IDX =  LUT_INPUT_ORDER.index(V_INT)
#
##the order of inputs is the same in simulator and LUT
#LUT_TO_INPUT_IDX = [LUT_INPUT_ORDER.index(i) for i in INPUT_NAMES]

LUT_INPUT_SHAPE = (5, 10, 39, 41, 41, 12, 12)


INPUT_LEN_2D = 6
#INPUT_LEN_3D = 7

INPUT_NAMES_2D = [OMEGA,      RHO, THETA, PSI, V_OWN, V_INT]
#INPUT_NAMES_3D = [OMEGA, TAU, RHO, THETA, PSI, V_OWN, V_INT]


INPUT_SHAPE_2D = (5,     39, 41, 41, 12, 12)
#INPUT_SHAPE_3D = (5, 10, 39, 41, 41, 12, 12)

INPUT_UNITS_2D = ["idx",      "ft", "rad", "rad", "ft/s", "ft/s"]
#INPUT_UNITS_3D = ["idx", "s", "ft", "rad", "rad", "ft/s", "ft/s"]

# 5 angular velocities (corresponding to the last action)
#OMEGA_DISCRETIZATION = [OMEGA_SL, OMEGA_WL, OMEGA_COC, OMEGA_WR, OMEGA_SR]
LUT_OMEGA_DISCRETIZATION = [OMEGA_COC, OMEGA_WR, OMEGA_WL, OMEGA_SR, OMEGA_SL]
# 12 speed values, non-linearly disposed   (ft/s)
LUT_V_INT_DISCRETIZATION = [0.0, 100.0, 142.0, 203.0, 290.0, 414.0, 545.0, 676.0, 807.0, 938.0, 1069.0, 1200.0]
# 12 other speed values, non-linearly disposed   (ft/s)
LUT_V_OWN_DISCRETIZATION = [50.0, 87.6, 139.9, 211.0, 307.2, 438.9, 570.6, 702.4, 834.1, 965.8, 1097.5, 1200.0]
#41 angles (rad) from -pi to +pi, linearly disposed, rounded 4
LUT_PSI_DISCRETIZATION = [-3.1416, -2.9845, -2.8274, -2.6704, -2.5133, -2.3562, -2.1991, -2.042, -1.885, -1.7279, -1.5708, -1.4137, -1.2566, -1.0996, -0.9425, -0.7854, -0.6283, -0.4712, -0.3142, -0.1571, 0.0, 0.1571, 0.3142, 0.4712, 0.6283, 0.7854, 0.9425, 1.0996, 1.2566, 1.4137, 1.5708, 1.7279, 1.885, 2.042, 2.1991, 2.3562, 2.5133, 2.6704, 2.8274, 2.9845, 3.1416]
#"psi": np.linspace(-math.pi, +math.pi, 41)
#41 angles (rad) from -pi to +pi, linearly disposed, rounded 4
LUT_THETA_DISCRETIZATION = [-3.1416, -2.9845, -2.8274, -2.6704, -2.5133, -2.3562, -2.1991, -2.042, -1.885, -1.7279, -1.5708, -1.4137, -1.2566, -1.0996, -0.9425, -0.7854, -0.6283, -0.4712, -0.3142, -0.1571, 0.0, 0.1571, 0.3142, 0.4712, 0.6283, 0.7854, 0.9425, 1.0996, 1.2566, 1.4137, 1.5708, 1.7279, 1.885, 2.042, 2.1991, 2.3562, 2.5133, 2.6704, 2.8274, 2.9845, 3.1416]
#"theta": np.linspace(-math.pi, +math.pi, 41)
#39 distances (range)   (ft)
LUT_RHO_DISCRETIZATION = [   499.,    800.,   2508.,   4516.,   6525.,   8534.,  10543.,  12551.,  14560.,  
       16569.,  18577.,  20586.,  22595.,  24603.,  26612.,  28621.,  30630.,  32638.,
       34647.,  36656.,  38664., 40673.,  42682.,  44690.,  46699. , 48708.,  50717.,
       52725.,  54734.,  56743.,  58751.,  60760.,  75950.,  94178., 112406., 130634.,
      148862., 167090., 185318.]
#vertical separation parameter (tau), time to same altitude
LUT_TAU_DISCRETIZATION = [  0.,   1.,   5.,  10.,  20.,  40.,  60.,  80., 100., 101.]

LUT_DISCRETIZATION = {
                  OMEGA: LUT_OMEGA_DISCRETIZATION,
                  V_INT: LUT_V_INT_DISCRETIZATION,
                  V_OWN: LUT_V_OWN_DISCRETIZATION, 
                  PSI: LUT_PSI_DISCRETIZATION, 
                  THETA: LUT_THETA_DISCRETIZATION, 
                  RHO: LUT_RHO_DISCRETIZATION,
                 }

LUT_DISCRETE_VALUES_2D = [                  
                      LUT_OMEGA_DISCRETIZATION,
                      LUT_V_INT_DISCRETIZATION,
                      LUT_V_OWN_DISCRETIZATION, 
                      LUT_PSI_DISCRETIZATION, 
                      LUT_THETA_DISCRETIZATION, 
                      LUT_RHO_DISCRETIZATION,
                     ]
   
LUT_DISCRETE_VALUES_3D = [                  
                      LUT_OMEGA_DISCRETIZATION,
                      LUT_TAU_DISCRETIZATION,      #3D includes vertical separation
                      LUT_V_INT_DISCRETIZATION,
                      LUT_V_OWN_DISCRETIZATION, 
                      LUT_PSI_DISCRETIZATION, 
                      LUT_THETA_DISCRETIZATION, 
                      LUT_RHO_DISCRETIZATION,
                     ]
               
        
###############################################################################

def _load_lut_npz(case='2d', mode='costs', folder=None):
   
   if folder is None:
      #folder = './lut/'
      folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lut')
   
   if mode=='costs': 
       if case == '2d':
          lut_file=os.path.join(folder,'lut_output_int16_tau_0.npz')
       else:
          lut_file=os.path.join(folder,'lut_output_int16.npz')
       
       lut = np.load(lut_file, allow_pickle=True, mmap_mode='r')
       lut = lut['lut']

   elif mode=='prefs':
       if case == '2d':
          lut_file=os.path.join(folder,'pref_lut_output_uint8_tau_0.npz')
       else:
          lut_file=os.path.join(folder,'pref_lut_output_uint8.npz')
       
       lut = np.load(lut_file, allow_pickle=True, mmap_mode='r')
       lut = lut['preflut']       
       
   else: # mode=='actions
       if case == '2d':
          lut_file=os.path.join(folder,'act_lut_output_uint8_tau_0.npz')
       else:
          lut_file=os.path.join(folder,'act_lut_output_uint8.npz')
       
       lut = np.load(lut_file, allow_pickle=True, mmap_mode='r')
       lut = lut['actlut']

   return lut

###############################################################################

def _get_lut_header(case='2d'):

   if case == '2d':
       return INPUT_SHAPE_2D, LUT_DISCRETE_VALUES_2D, INPUT_NAMES_2D
   else:
       return INPUT_SHAPE_3D, LUT_DISCRETE_VALUES_3D, INPUT_NAMES_3D


###############################################################################

def get_input_values_from_flat_index(flt_idx:int, shape:tuple=INPUT_SHAPE_2D, discrete_values:list=LUT_DISCRETE_VALUES_2D):
   
   mlt_idx = np.unravel_index(flt_idx, shape)
   input_arr = np.array([discrete_values[i][mlt_idx[i]] for i in range(len(shape))] )
   return input_arr


###############################################################################

def get_input_values_from_multi_index(multi_index, discrete_values:list=LUT_DISCRETE_VALUES_2D):
   input_values = np.array([discrete_values[i][multi_index[i]] for i in range(len(discrete_values))] )
   return input_values

###############################################################################

def get_flat_index(multi_index, shape:tuple=INPUT_SHAPE_2D):
    return np.ravel_multi_index(multi_index, shape)


def get_multi_index(flat_index, shape:tuple=INPUT_SHAPE_2D):
    return np.unravel_index(flat_index, shape)

###############################################################################

def get_multi_index_iterator(shape:tuple=INPUT_SHAPE_2D):
   return product( *map(range, shape) )

###############################################################################

def nearest_multi_index(input_values, discrete_values:list=LUT_DISCRETE_VALUES_2D):   #input_values = [last_a, rho, theta, psi, v_own, v_int]
    multi_index = tuple([np.abs(np.array(discrete_values[i])-input_values[i]).argmin() for i in range(len(discrete_values))])
    return multi_index

###############################################################################

def get_costs(input_values, lut, discrete_values:list=LUT_DISCRETE_VALUES_2D):   #input_values = [last_a, rho, theta, psi, v_own, v_int]
    multi_index = nearest_multi_index(input_values, discrete_values)
    return lut[multi_index]

def get_action_from_costs(input_values, lut, discrete_values:list=LUT_DISCRETE_VALUES_2D):   #input_values = [last_a, rho, theta, psi, v_own, v_int]
    return np.argmin(get_costs(input_values, lut, discrete_values))


###############################################################################


class NearestLUT(AbstractModel) :
   
    def __init__(self, verbose=False, max_distance=np.inf, invert_L_R=True, mode='actions', case='2d'):    # max_distance = 68000 ft
    
        #case horizontal ('2d') or complete ('3d')
        case=case
        
        #mode = [costs, prefs, actions] 
        # prefs --> using the file containing the action preferences (order)
        self.mode = mode

        self.lut = _load_lut_npz(case=case, mode=mode)
        
        self.max_distance = max_distance
        if verbose:
            print(len(self.lut), "rows loaded.")
            print("shape:", self.lut.shape)
            print("dtype:", self.lut.dtype)
            #print("max:", self.lut.max())
            #print("min:", self.lut.min())        
        self.shape, self.discrete_values, self.names = _get_lut_header(case=case)
        self.invert_L_R = invert_L_R

    def get_flat_index(self, multi_index):
        return np.ravel_multi_index(multi_index, self.shape)
        
    def get_multi_index(self, flat_index):
        return np.unravel_index(flat_index, self.shape)
    
    def get_multi_index_iterator(self):
        return product( *map(range, self.shape) )
    
    def get_values_from_multi_index(self, mlt_idx):
        #return np.array([self.discrete_values[name][mlt_idx[i]] for i, name in enumerate(self.names)] )
        return np.array([self.discrete_values[i][mlt_idx[i]] for i in range(len(self.shape))] )

    def get_observation_from_multi_index(self, mlt_idx):
        #return HorizontalObservation(**{name: self.discrete_values[name][mlt_idx[i]] for i, name in enumerate(self.names)})
        return HorizontalObservation(self.get_values_from_multi_index(mlt_idx))
        
    def get_values_from_flat_index(self, flt_idx:int):
        mlt_idx = self.get_multi_index(flt_idx)
        return self.get_input_from_multi_index(mlt_idx)

    def get_encounter_from_flat_index(self, flt_idx):
        mlt_idx = self.get_multi_index(flt_idx)
        return self.get_encounter_from_multi_index(mlt_idx)

    def nearest_multi_index(self, obs):
        multi_index = tuple([np.abs(np.array(self.discrete_values[i])-obs[i]).argmin() for i in range(len(self.shape))])
        return multi_index

    def nearest_obs(self, obs):
        multi_index = self.nearest_multi_index(obs)
        #obs = HorizontalObservation(**{name:self.discrete_values[name][multi_index[i]] for i, name in enumerate(self.names)})
        return self.get_observation_from_multi_index(multi_index)

    def nearest_index(self, obs):
        multi_index = self.nearest_multi_index(obs)
        index = self.get_flat_index(multi_index)
        return index
    
    def predict(self,
                obs=None, *,
                last_a=None, 
                rho=None, theta=None, psi=None, v_own=None, v_int=None):
                #tau=None):
                    
        #enc = ensure_horizontal_encounter(obs, last_a=last_a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int, tau=tau)
        obs = HorizontalObservation(obs, last_a=last_a, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)

        if obs.rho > self.max_distance:
            #print(self.max_distance)
            return np.array([0., +1., +1., +1., +1.])
        
        else:

            if self.invert_L_R:
                obs.last_a = [0, 2, 1, 4, 3][obs.last_a]

            multi_index = self.nearest_multi_index(obs)
            
            if self.mode == 'actions':
                action = self.lut[multi_index]
                costs = np.repeat(+1.0, 5)
                costs[action] = 0.0
            else:
                costs = self.lut[multi_index]
            
            #inverting R and L...
            if self.invert_L_R:
                costs = costs[[0,2,1,4,3]]
            
            return costs




###############################################################################


if __name__ == '__main__' :
   
    from acasxu_basics import ACT_NAMES
    
    print()
    print("TESTING LOAD LUT NPZ FILE...")
    print()

    print("2D case")
    print("Loading...")
    lut_2d = _load_lut_npz(case='2d')
    print("shape:", lut_2d.shape)
    print("dtype:", lut_2d.dtype)
    print("max:", lut_2d.max())
    print("min:", lut_2d.min())
    #print("max':", lut_2d.max(axis=0))
    #print("min':", lut_2d.min(axis=0))

    print()
    print("The 2d case input discretization is:")
    shape_2d, discrete_dict_2d, names_2d = _get_lut_header(case='2d')
 
    print("shape:", shape_2d)
    print("names:", names_2d)
    total_rows = np.prod(shape_2d)
    print(total_rows, "values")

    print("2D case - actions")
    print("Loading...")
    act_lut = _load_lut_npz(case='2d', mode='actions')

    print()
    print("Counting...")
    #number_COC = np.count_nonzero(lut_2d.argmin(axis=-1)==ACT_COC)
    number_COC = np.count_nonzero(act_lut==LUT_COC_IDX)
    print("COC:", number_COC, round(number_COC/total_rows*100,2), "%")
    #number_other = np.count_nonzero(lut_2d.argmin(axis=-1)!=ACT_COC)
    number_other = np.count_nonzero(act_lut!=LUT_COC_IDX)
    print("other:", number_other, round(number_other/total_rows*100,2), "%")
    
    print()
    number_WR = np.count_nonzero(act_lut==LUT_WR_IDX)
    print("WR:", number_WR, round(number_WR/total_rows*100,2), "%")
    number_WL = np.count_nonzero(act_lut==LUT_WL_IDX)
    print("WL:", number_WL, round(number_WL/total_rows*100,2), "%")
    number_SR = np.count_nonzero(act_lut==LUT_SR_IDX)
    print("SR:", number_SR, round(number_SR/total_rows*100,2), "%")
    number_SL = np.count_nonzero(act_lut==LUT_SL_IDX)
    print("SL:", number_SL, round(number_SL/total_rows*100,2), "%")

    print()
    print("Counting when rho > 60760")
    RHO_60760_IDX = np.argwhere(np.array(LUT_DISCRETE_VALUES_2D[1]) >= 60760)[0][0]  #rho is index 1
    print(RHO_60760_IDX, LUT_DISCRETE_VALUES_2D[1][RHO_60760_IDX])
    number_COC = np.count_nonzero(act_lut[:,RHO_60760_IDX:,:,:,:,:]==LUT_COC_IDX)
    number_other = np.count_nonzero(act_lut[:,RHO_60760_IDX:,:,:,:,:]!=LUT_COC_IDX)
    total_rows = number_other + number_COC
    print("other:", number_other, round(number_other/total_rows*100,2), "%")
    print("COC:", number_COC, round(number_COC/total_rows*100,2), "%")
    print()
    print("TESTING LUT MODEL...")
    print()

    lut = NearestLUT()

    #simple test
    last_a = 0 #coc
    rho, theta, psi, v_own, v_int = 10543.0, 2.6704, 0.9425, 834.1, 545.0
    #rho, theta, psi, v_own, v_int = 10000., +2.7, +1., 800., 600.
    #rho, theta, psi, v_own, v_int = 499, -3.1416, -3.1416, 50., 0.      
    #state = [last_command, rho, theta, psi, v_own, v_int]
    #state = [last_command, v_int, v_own, psi, theta, rho]
    #dic_state = {name:v for name, v in zip(lut.names, state)}
   
    obs = HorizontalObservation(last_a=last_a, v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)

    print("INPUT:", obs)
    print("distance (ft)", rho)
    print("own speed (ft/s)", round(v_own, 2))
    print("intruder speed (ft/s)", round(v_int, 2))
    print("initial relative angle (degrees)", round(theta * 180 / math.pi, 2))
    print("intruder cap angle (degrees)", round(psi * 180 / math.pi, 2))

    print()
    nearest_i = lut.nearest_index(obs)
    print("nearest index=", nearest_i)
    nearest_m = lut.nearest_multi_index(obs)
    print("nearest mltid=", nearest_m)
    nearest_i2 = lut.get_flat_index(nearest_m)
    print("nearest index (verif)=", nearest_i2)
    nearest_m2 = lut.get_multi_index(nearest_i)
    print("nearest mltid (verif)=", nearest_m2)
    nearest_v = lut.nearest_obs(obs)
    print("nearest values=", nearest_v)
    enc2 = lut.get_observation_from_multi_index(nearest_m)
    print("INPUT (verif):", enc2)

    res = lut.predict(obs)
    print("OUTPUT (q-values):", res)
    command = np.argmin(res)
    print("action:", command, LUT_ACT_ORDER[command])     


    print()
    lut = NearestLUT(mode='costs')

    last_a = 0 #coc
    #rho, theta, psi, v_own, v_int = +10000., +2.7, +1., 800., 600.
    rho, theta, psi, v_own, v_int = 10543.0, 2.6704, 0.9425, 834.1, 545.0
    obs = HorizontalObservation( last_a=last_a, v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)
    print("REFERENCE INPUT:", obs)
    res = lut.predict(obs)
    print("OUTPUT (q-values):", res)
    command = np.argmin(res)
    print("action:", command, LUT_ACT_ORDER[command])     
    obs.theta = - obs.theta
    obs.psi = - obs.psi
    print("SYMMETRICAL INVERSE INPUT:", obs)
    res = lut.predict(obs)
    print("OUTPUT (q-values):", res)
    command = np.argmin(res)
    print("action:", command, LUT_ACT_ORDER[command])     


















