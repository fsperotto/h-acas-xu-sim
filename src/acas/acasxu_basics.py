# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:05:50 2024
Last Modified on 2025 September

@author: fperotto

Basic functions, enumerations, constants, and classes to modelize the Horizontal ACAS-Xu simulation.
"""

###############################################################################

import numpy as np
import math


###############################################################################
# CONSTANTS
###############################################################################

#conversion m to ft
FT_TO_M_FACTOR = 0.3048             #exact size of a feet in meters, international convention
M_TO_FT_FACTOR = 1.0/FT_TO_M_FACTOR #size of a meter in feets

#GOLDEN_RATIO = (math.sqrt(5)-1)/2   #golden ratio

#smallest value to be considered like zero   (replaces a==0 with abs(a)<TOL_ZERO)
TOL_ZERO = 1e-14

#tolerated error on time (for NMAC or CPA)
TOL_TIME  = 0.1 # 100ms

#tolerated error on distance (for NMAC or CPA)
TOL_DIST  = 1.0 # 1m

#time and space units
DISTANCE_UNIT_STR = "ft"
TIME_UNIT_STR = "s"

#distance to consider a near mid-air colision
DEFAULT_NMAC_DIST = 500.0   #500ft

DEFAULT_MAX_ACAS_DISTANCE = 68000.0  #68000ft

DEFAULT_MAX_EPISODE_TIME = 300.0  #  300s -> limiting to 5 minutes flight

DEFAULT_MIN_SPEED = 50.0    #ft/s
DEFAULT_MAX_SPEED = 1200.0  #ft/s

###############################################################################
## DISCRETE ACTION CONSTANTS
###############################################################################

SL = "SL"
WL = "WL"
COC = "COC"
WR = "WR"
SR = "SR"

ACT_NAMES = (SL, WL, COC, WR, SR)

ACT_LEN = len(ACT_NAMES)    # 5

ACT_DESCRIPTIONS = ('strong-left', 'weak-left', 'clear-of-conflict', 'weak-right', 'strong-right')

SL_IDX  = ACT_NAMES.index(SL)   # 0 # strong left
WL_IDX  = ACT_NAMES.index(WL)   # 1 # weak left
COC_IDX = ACT_NAMES.index(COC)  # 2 # clear of conflict, no action
WR_IDX  = ACT_NAMES.index(WR)   # 3 # weak right
SR_IDX  = ACT_NAMES.index(SR)   # 4 # strong right

#ACT_INDEXES = (0, 1, 2, 3, 4)

DIC_ACT = {
                SL_IDX : ('l', 'L', 'sl', 'SL', 'left', 'LEFT', 'strong-left', 'STRONG-LEFT', SL_IDX),
                WL_IDX : ('wl', 'WL', 'weak-left', 'WEAK-LEFT', WL_IDX),
                COC_IDX: ('coc', 'COC', 
                          'clear-of-conflict', 'clear of conflict', 
                          'clear', 'none', 'CLEAR', 'NONE', 
                          'CLEAR-OF-CONFLICT', 'CLEAR OF CONFLICT', COC_IDX),
                WR_IDX : ('wr', 'WR', 'weak-right', 'WEAK-RIGHT', WR_IDX),
                SR_IDX : ('r', 'R', 'sr', 'SR', 'right', 'RIGHT', 'strong-right', 'STRONG-RIGHT', SR_IDX)
              }

OMEGA_SL = math.radians(+3.0)
OMEGA_WL = math.radians(+1.5)
OMEGA_COC = 0.0
OMEGA_WR = math.radians(-1.5)
OMEGA_SR = math.radians(-3.0)

ACT_OMEGA = (OMEGA_SL, OMEGA_WL, OMEGA_COC, OMEGA_WR, OMEGA_SR)

#------------------------------------------------------------------------------

#indax of an action given its name
def get_act_idx(name:str, default_value:int=None) -> int:
    """
    Returns the action index given its name. If the name is not known, returns a default value.

    Args:
        name (str): the name of the action (e.g. "weak left", "strong right", "clear of conflict".

    Returns:
        float: the same angle but in the interval [-pi, +pi].
    """
    
    #if name correspond to some action
    for idx, names in DIC_ACT.items():
      if name in names:
        return idx
    #otherwise
    return default_value

###############################################################################

def omega_to_act(omega:float, act_angles=ACT_OMEGA) -> int:
    """
    Return the index of the discrete action given angular speed.
    """
    #get the difference of each angle to omega
    diff = tuple(abs(rad_mod(angle-omega)) for angle in act_angles)
    #returns the argmin
    return diff.index(min(diff))
    
###############################################################################
## OBSERVATION AND ENCOUNTER CONSTANTS
###############################################################################

OMEGA = "omega"
RHO = "rho"
THETA = "theta"
PSI = "psi"
V_OWN = "v_own"
V_INT = "v_int"

#------------------------------------------------------------------------------

ENC_NAMES = (RHO, THETA, PSI, V_OWN, V_INT)
ENC_UNITS = ("ft", "rad", "rad", "ft/s", "ft/s")

ENC_LEN = len(ENC_NAMES)

ENC_RHO_IDX = ENC_NAMES.index(RHO)         # 1: rho, distance (ft) [0, 60760]
ENC_THETA_IDX = ENC_NAMES.index(THETA)     # 2: theta, angle to intruder relative to ownship heading (rad) [-pi,+pi]
ENC_PSI_IDX = ENC_NAMES.index(PSI)         # 3: psi, heading of intruder relative to ownship heading (rad) [-pi,+pi]
ENC_V_OWN_IDX = ENC_NAMES.index(V_OWN)     # 4: v_own, speed of ownship (ft/sec) [100, 1145? 1200?] 
ENC_V_INT_IDX = ENC_NAMES.index(V_INT)     # 5: v_int, speed in intruder (ft/sec) [0? 60?, 1145? 1200?] 

#------------------------------------------------------------------------------

OBS_NAMES = (OMEGA, RHO, THETA, PSI, V_OWN, V_INT)
OBS_UNITS = ("rad/s", "ft", "rad", "rad", "ft/s", "ft/s")

OBS_LEN = len(OBS_NAMES)

OBS_OMEGA_IDX = OBS_NAMES.index(OMEGA)     # 0: omega, current angular velocity (rad/s) [rad(-3deg), rad(+3deg)]   #corresponding to last action
OBS_RHO_IDX = OBS_NAMES.index(RHO)         # 1: rho, distance (ft) [0, 60760]
OBS_THETA_IDX = OBS_NAMES.index(THETA)     # 2: theta, angle to intruder relative to ownship heading (rad) [-pi,+pi]
OBS_PSI_IDX = OBS_NAMES.index(PSI)         # 3: psi, heading of intruder relative to ownship heading (rad) [-pi,+pi]
OBS_V_OWN_IDX = OBS_NAMES.index(V_OWN)     # 4: v_own, speed of ownship (ft/sec) [100, 1145? 1200?] 
OBS_V_INT_IDX = OBS_NAMES.index(V_INT)     # 5: v_int, speed in intruder (ft/sec) [0? 60?, 1145? 1200?] 

#------------------------------------------------------------------------------

HEADING_E = 0.0
HEADING_N = math.radians(+90.0)
HEADING_W = math.radians(+180.0)
HEADING_S = math.radians(+270.0)


###############################################################################

#radians between -pi and +pi
def rad_mod(angle_rad:float) -> float:
    """
    Returns the given angle or its equivalent in the range between -pi and +pi.

    Args:
        angle_rad (float): the angle in radians.

    Returns:
        float: the same angle but in the interval [-pi, +pi].
    """
    return ( (angle_rad + math.pi) % (math.pi*2) ) - math.pi

###############################################################################

#from polar coordinates to cartesian coordinates
def cartesian_coords(d:float, angle_rad:float) -> tuple:    #tuple(float)
    """
    Returns the correponding (x, y) cartesian coordinates given the polar coordinates (rho, theta), i.e. distance (vector length) and angle in radians.

    Args:
        d (float): the distance or vector length, or radius.
        angle_rad (float): phi, the angle in radians.

    Returns:
        tuple(float, float): corresponding (x, y) cartesian coordinates.
    """
    #x = d * math.cos(angle_rad)
    #y = d * math.sin(angle_rad)
    #return (x, y)
    return  ( d * math.cos(angle_rad),  d * math.sin(angle_rad) )
    
#------------------------------------------------------------------------------

#from cartesian coordinates to polar coordinates
def polar_coords(x:float, y:float) -> tuple:    #tuple(float)
    """
    Returns the correponding (distance, angle in radians) polar coordinates given the cartesian coordinates (x, y).

    Args:
        x (float): the x cartesian coordinate.
        y (float): the y cartesian coordinate.

    Returns:
        tuple(float, float): corresponding (d, theta_rad) polar coordinates.
    """
    #d     = math.hypot(x, y)  # relative distance
    #theta = rad_mod(math.atan2(y, x))   #relative positional angle
    #return (d, angle)
    return  math.hypot(x, y),  rad_mod(math.atan2(y, x))

###############################################################################

def projected_position(x:float, y:float, heading:float, v:float, omega:float=0.0, t:float=1.0) -> tuple:    #tuple(float, float, float)    
    """
    Returns the future position in cartesian coordinates and heading angle (in radians), given the current position (x, y), 
    current heading, tangetial and angular velocities, and time.
    When omega is zero, it corresponds to a uniform regular linear motion. 
    When omega is different from zero, it is a uniform circular motion.

    Args:
        x (float): the current x_0 cartesian coordinate.
        y (float): the current y_0 cartesian coordinate.
        heading (float): current heading in radians.
        v (float) : tangetial speed.
        omega (float) : angular speed.
        t (float) : elapsed time to future position.

    Returns:
        tuple(float, float, float): corresponding (x_t, y_t, heading_t).
    """
 
    #omega is zero or very close to zero
    if abs(omega) < TOL_ZERO:  
    
        #linear motion:
        ##calculate total traveled distance
        #d = t * v   
        ##separate x and y components
        #x_t = x + (d * math.cos(heading))
        #y_t = y + (d * math.sin(heading))
        
        #return (x_t, y_t, heading)
        return (x + v * math.cos(heading) * t ,  y + v * math.sin(heading) * t , heading)
        
    else:
    
        ##polar equivalence
        ##R = v / abs(omega)    #rayon
        ##xc = x - R*math.sin(heading)*math.copysign(1, omega)   #center x
        ##yc = y + R*math.cos(heading)*math.copysign(1, omega)   #center y
        #closed-form arc displacement
        #x_t = x + (v / omega) * (math.sin(heading + omega * t) - math.sin(heading))
        #y_t = y - (v / omega) * (math.cos(heading + omega * t) - math.cos(heading))
        #heading_t = heading + omega * t
        
        #return (x_t, y_t, heading_t)
        return ( x + (v / omega) * (math.sin(heading + omega * t) - math.sin(heading)),  #x
                 y - (v / omega) * (math.cos(heading + omega * t) - math.cos(heading)),  #y
                 heading + omega * t )                                                   #heading
               
###############################################################################
# AIRPLANE DATA TO TUPLE
#------------------------

def get_airplane_ulm_data(airplane) -> tuple:
    """
    get an object, dictionary, list, or np array and return the tuple (x, y, heading, v).
    """
    if isinstance(airplane, dict):
       return ( airplane['x'], airplane['y'], airplane['heading'], airplane['v'])
    elif isinstance(airplane, list) or isinstance(airplane, tuple) or isinstance(airplane, np.ndarray):
       return airplane[:4]
    else:
       return (airplane.x, airplane.y, airplane.heading, airplane.v)

#--------------------------------------------------------------------------

def get_airplane_data(airplane) -> tuple:
    """
    get an object, dictionary, list, or np array and return the tuple (x, y, heading, v, omega).
    """
    if isinstance(airplane, dict):
       return ( airplane['x'], airplane['y'], airplane['heading'], airplane['v'], airplane['omega'] )
    elif isinstance(airplane, list) or isinstance(airplane, tuple) or isinstance(airplane, np.ndarray):
       return airplane[:5]
    else:
       return (airplane.x, airplane.y, airplane.heading, airplane.v, airplane.omega)


###############################################################################
# ENCOUNTER DATA TO TUPLE
#--------------------------

def get_encounter_data(data) -> tuple:
    """
    Get a dictionary, list, tuple, array, or encounter-like object and return an iterable containing: rho, theta, psi, v_own, v_int.
    If the input is a dictionary, then the attibutes are searched by name, and the result is a tuple.
    If the input is a list, then the attibutes are supposed to be given in the correct positions, and the result is a list.
    If the input is a ndarray, then the attibutes are supposed to be given in the correct positions, and the result is a ndarray.
    Note that ndarrays can have batch supplementary dimensions.
    If the input is an object, then the attibutes are searched by name, and the result is a tuple.
    """
    attributes = ('rho','theta','psi','v_own','v_int')
    if isinstance(data, dict):
       return (data[k] for k in attributes) 
    elif isinstance(data, list) or isinstance(encounter, tuple):
       return data[:5]
    elif isinstance(data, np.ndarray):
       return data[..., :5]
    else:
       #return (data.x, data.y, data.heading, data.v, data.omega)
       return (getattr(data, k) for k in attributes)

#--------------------------------------------------------------------------

def get_encounter_dict(encounter) -> tuple:
    """
    Get a dictionary, list, tuple, array, or encounter-like object and return a dictionary: {'rho':rho, 'theta':theta, 'psi':psi, 'v_own':v_own, 'v_int':v_int}.
    If the input is a dictionary, then the attibutes are searched by name..
    If the input is a list or ndarray, then the attibutes are supposed to be given in the correct positions.
    Note that ndarrays can have batch supplementary dimensions.
    If the input is an object, then the attibutes are searched by name.
    """
    attributes = ('rho','theta','psi','v_own','v_int')
    if isinstance(data, dict):
       return {k:data[k] for k in attributes}
    elif isinstance(data, list) or isinstance(data, tuple):
       return {k:data[i] for i, k in enumerate(attributes)} 
    elif isinstance(data, np.ndarray):
       return {k:data[..., i] for i, k in enumerate(attributes)} 
    else:
       return {k:getattr(data, k) for k in attributes} 
    #else:
    #   raise ValueError("[ERROR] Incomplete observation passed as an array.")

#--------------------------------------------------------------------------

def get_encounter_data_from_airplanes(own, intruder):
    
    #speeds from aisplane speeds
    v_own = own.v
    v_int = intruder.v
    
    #displace all and make own the center
    rho_x, rho_y = intruder.x - own.x, intruder.y - own.y
    rho, theta = polar_coords(rho_x, rho_y)
    #rotate all to neutralize own.heading
    theta, psi = rad_mod(theta-own.heading), rad_mod(intruder.heading-own.heading)
    
    return (rho, theta, psi, v_own, v_int)

#--------------------------------------------------------------------------

def get_encounter_dict_from_airplanes(own, intruder):
    attributes = ('rho','theta','psi','v_own','v_int')
    values = get_encounter_data_from_airplanes(own, intruder)
    return {k:v for k, v in zip(attributes, values)}


###############################################################################
# OBSERVATION DATA TO TUPLE
#---------------------------

def get_observation_data(data) -> tuple:
    """
    Get a dictionary, list, tuple, array, or observation-like object and return an iterable containing: omega, rho, theta, psi, v_own, v_int.
    If the input is a dictionary, then the attibutes are searched by name, and the result is a tuple.
    If the input is a list, then the attibutes are supposed to be given in the correct positions, and the result is a list.
    If the input is a ndarray, then the attibutes are supposed to be given in the correct positions, and the result is a ndarray.
    Note that ndarrays can have batch supplementary dimensions.
    If the input is an object, then the attibutes are searched by name, and the result is a tuple.
    """
    attributes = ('omega','rho','theta','psi','v_own','v_int')
    if isinstance(data, dict):
       return (data[k] for k in attributes) 
    elif isinstance(data, list) or isinstance(data, tuple):
       return data[:6]
    elif isinstance(data, np.ndarray):
       return data[..., :6]
    else:
       #return (data.x, data.y, data.heading, data.v, data.omega)
       return (getattr(data, k) for k in attributes)

#--------------------------------------------------------------------------

def get_observation_dict(data) -> tuple:
    """
    Get a dictionary, list, tuple, array, or observation-like object and return a dictionary: {'rho':rho, 'theta':theta, 'psi':psi, 'v_own':v_own, 'v_int':v_int}.
    If the input is a dictionary, then the attibutes are searched by name..
    If the input is a list or ndarray, then the attibutes are supposed to be given in the correct positions.
    Note that ndarrays can have batch supplementary dimensions.
    If the input is an object, then the attibutes are searched by name.
    """
    attributes = ('omega','rho','theta','psi','v_own','v_int')
    if isinstance(data, dict):
       return {k:data[k] for k in attributes}
    elif isinstance(data, list) or isinstance(data, tuple):
       return {k:data[i] for i, k in enumerate(attributes)} 
    elif isinstance(data, np.ndarray):
       return {k:data[..., i] for i, k in enumerate(attributes)} 
    else:
       return {k:getattr(data, k) for k in attributes} 
    #else:
    #   raise ValueError("[ERROR] Incomplete observation passed as an array.")

#--------------------------------------------------------------------------

def get_observation_data_from_airplanes(own, intruder):
    
    #speeds from aisplane speeds
    v_own = own.v
    v_int = intruder.v
    
    #displace all and make own the center
    rho_x, rho_y = intruder.x - own.x, intruder.y - own.y
    rho, theta = polar_coords(rho_x, rho_y)
    #rotate all to neutralize own.heading
    theta, psi = rad_mod(theta-own.heading), rad_mod(intruder.heading-own.heading)
    #last action (current angular velocity)
    omega=own.omega
    
    return (omega, rho, theta, psi, v_own, v_int)

#--------------------------------------------------------------------------

def get_observation_dict_from_airplanes(own, intruder):
    attributes = ('omega','rho','theta','psi','v_own','v_int')
    values = get_observation_data_from_airplanes(own, intruder)
    return {k:v for k, v in zip(attributes, values)}

#--------------------------------------------------------------------------


###############################################################################
# ENCOUNTER
#-----------

def projected_separation(a1=None, a2=None, *,
                                x1:float=None, y1:float=None, h1:float=None, v1:float=None, w1:float=0.0,
                                x2:float=None, y2:float=None, h2:float=None, v2:float=None, w2:float=0.0,
                                t:float=1.0) -> tuple :

    #get data from airplane object if given
    if a1 is not None:
        x1, y1, h1, v1, w1 = get_airplane_data(a1)
    if a2 is not None:
        x2, y2, h2, v2, w2 = get_airplane_data(a2)

    #calculate positions and distance at given t
    p1_t = (x1_t, y1_t, h1_t) = projected_position(x=x1, y=y1, heading=h1, v=v1, omega=w1, t=t)
    p2_t = (x2_t, y2_t, h2_t) = projected_position(x=x2, y=y2, heading=h2, v=v2, omega=w2, t=t)   
    d_t = math.dist(p1_t, p2_t)
    
    return (d_t, p1_t, p2_t)
    
#--------------------------------------------------------------------------

#closest point of approach if linear trajectories and uniform regular motion
def find_cpa_ulm_ulm(a1=None, a2=None, *,
                     x1:float=None, y1:float=None, h1:float=None, v1:float=None, 
                     x2:float=None, y2:float=None, h2:float=None, v2:float=None):
    """
    Compute Closest Point of Approach (CPA) between two aircraft,
    both flying in straight lines at constant velocity (uniform linear motion).

    Args:
        x1, y1, h1, v1, w1 : situation of aircraft1, i.e. x, y coordinates, heading angle (in radians), linear or tangential speed, omega (angular speed)
        x2, y2, h2, v2, w2 : situation of aircraft2.

    Returns:
        t_cpa (float): time to CPA (>=0)
        p1_cpa (np.ndarray): position of aircraft 1 at CPA
        p2_cpa (np.ndarray): position of aircraft 2 at CPA
        d_cpa (float): minimal separation distance
    """
    #get data from airplane object if given
    if a1 is not None:
        x1, y1, h1, v1 = get_airplane_ulm_data(a1)
    if a2 is not None:
        x2, y2, h2, v2 = get_airplane_ulm_data(a2)

    #relative velocities per cartesian component
    vx = v1 * math.cos(h1) - v2 * math.cos(h2)  #relative velocity on x 
    vy = v1 * math.sin(h1) - v2 * math.sin(h2)  #relative velocity on y
    
    #approximation factor (squared relative v)
    relative_v_sqr = vx*vx + vy*vy   #vx^2 + vy^2
    
    #parallel flight
    if abs(relative_v_sqr) < TOL_ZERO:

        t_cpa = 0.0
        #positions at CPA are initial ones
        p1_cpa = (x1, y1, h1)
        p2_cpa = (x2, y2, h1)

    #non-paralell flight
    else:

        #relative distance per cartesian component
        dx = x1 - x2
        dy = y1 - y2
        #solve cpa equation
        t_cpa = -(dx * vx + dy * vy) / relative_v_sqr
        t_cpa = max(0.0, t_cpa)
        #positions at CPA
        p1_cpa = projected_position(x=x1, y=y1, heading=h1, v=v1, t=t_cpa)
        p2_cpa = projected_position(x=x2, y=y2, heading=h2, v=v2, t=t_cpa)   
    
    #minimal distance
    d_cpa = math.dist(p1_cpa, p2_cpa)    
    
    return t_cpa, p1_cpa, p2_cpa, d_cpa
    
#--------------------------------------------------------------------------

#closest point of approach if linear trajectories and uniform regular motion using numpy
def np_find_cpa_ulm(airplanes):
    """
    Compute Closest Point of Approach (CPA) between aircraft,
    all flying in straight lines at constant velocity (uniform linear motion).

    Args:
        airplanes: airplanes with attributes (x, y, heading, v)

    Returns:
        t_cpa (float): time to CPA (>=0)
        p1_cpa (np.ndarray): position of aircraft 1 at CPA
        p2_cpa (np.ndarray): position of aircraft 2 at CPA
        d_cpa (float): minimal separation distance
    """
    
    #cpa using numpy
    #----------------
     
    #initial positions
    p1 = np.array([a1.x, a1.y], dtype=float)
    p2 = np.array([a2.x, a2.y], dtype=float)
    
    #velocity component vectors
    v1 = a1.v * np.array([np.cos(a1.heading), np.sin(a1.heading)], dtype=float)
    v2 = a2.v * np.array([np.cos(a2.heading), np.sin(a2.heading)], dtype=float)
    
    #relative motion
    relative_pos = p1 - p2
    relative_v = v1 - v2
    
    #squared relative v
    relative_v_sqr = np.dot(relative_v, relative_v)
    
    #parallel flight
    if relative_v_sqr == 0.0:
        t_cpa = 0.0
    else:
        t_cpa = -(np.dot(r, v) / relative_v_sqr)
        t_cpa = max(0.0, t_cpa)
    
    # Positions at CPA
    p1_cpa = p1 + v1 * t_cpa
    p2_cpa = p2 + v2 * t_cpa
    
    # Minimal distance
    d_cpa = np.linalg.norm(p1_cpa - p2_cpa)

    return t_cpa, p1_cpa, p2_cpa, d_cpa
    
#--------------------------------------------------------------------------

##############################################################################################
# nmac incident functions
# -------------------

def find_nmac_ulm_ulm(a1=None, a2=None, *,
                      x1:float=None, y1:float=None, h1:float=None, v1:float=None, 
                      x2:float=None, y2:float=None, h2:float=None, v2:float=None, 
                      nmac_distance:float=DEFAULT_NMAC_DIST, 
                      t_max:float=DEFAULT_MAX_EPISODE_TIME
                     ) -> tuple :
    """
    Find next time when the horizontal separation of two straight-motion (ULM) aircraft becomes less or equal than nmac_distance (nmac_distance).

    Args:
        x1, y1, h1, v1 : situation of aircraft1 if a1 is None. If a1 is given, those values are ignored.
        x2, y2, h2, v2 : situation of aircraft1 if a1 is None. If a2 is given, those values are ignored.
        nmac_distance: threshold or incident distance (positive).
        t_max: maximum time horizon to consider. None for infinite (ubounded horizon).
        
    Returns:
        (t, (x1_t, y1_t), (x2_t, y2_t) where
            t (float): incident time >= 0, when distance first becomes <= eps
            (x1_t, y1_t), (x2_t, y2_t) (tuples): positions (x,y) of aircraft 1 and 2 at incident time t
        or None if no incident occurs in the interval [0, t_max].

    Notes:
        - If they are already within eps at the initial time, returns t=0 and initial positions.
    """
    #get data from airplane object if given
    if a1 is not None:
        x1, y1, h1, v1 = get_airplane_ulm_data(a1)
    if a2 is not None:
        x2, y2, h2, v2 = get_airplane_ulm_data(a2)
   
    #initial distance
    d = math.dist((x1,y1),(x2,y2))
    
    #initial relative distance per cartesian component
    dx = x1 - x2
    dy = y1 - y2
    #psi = a1.heading - a2.heading

    #relative velocity per cartesian component
    vx = v1 * math.cos(h1) - v2 * math.cos(h2)    #relative velocity on x 
    vy = v1 * math.sin(h1) - v2 * math.sin(h2)    #relative velocity on y

    #quadratic coefficients for |r0 + v_rel*t|^2 = nmac_distance^2
    a = vx*vx + vy*vy                    #coef_quadratic --> magnitude of the relative velocity squared    (faster than vx**2 + vy**2)

    #relative velocity zero --> never changes distance
    if a == 0:
        return None, (None, None, None), (None, None, None), None

    c = dx*dx + dy*dy - nmac_distance*nmac_distance  #coef_constant  --> margin distance to incident considering the initial positions  (faster than dx**2 + dy**2 - nmac_distance**2)

    #initial margin less than nmac_distance --> constraint violation from the beginning
    if c <= 0:
        return 0.0, (x1, y1, h1), (x2, y2, h2), d

    b = 2*(dx*vx + dy*vy)  #coef_linear -->  relative velocity projection along the initial separation vector
    
    #calculate discriminant (delta on the quadratic equation)
    delta = b*b - 4*a*c        #delta:  quadradic equation discriminant = b^2-4ac
    
    # no real solution --> never within nmac_distance        
    if delta < 0:
        return None, (None, None, None), (None, None, None), None
    
    #get sqrt(delta)
    sqrt_delta = math.sqrt(delta)     

    #two candidate times
    #quadratic solutions : (-b +/- sqrt(delta))/2a
    t1 = (-b-sqrt_delta) / (2*a)
    t2 = (-b+sqrt_delta) / (2*a)

    #two equivalent times : the incitend distance is reached in tangency, only once, then equivalent to the minimal distance time

    #keep only non-negative times within t_max    (negative t means that the event is on the past trajectory.)
    t_candidates = [t for t in (t1, t2) if 0 <= t]
    if t_max is not None:
        t_candidates = [t for t in (t1, t2) if t <= t_max]
    
    #empty list, no candidates within the considered interval [0, t_max]
    if not t_candidates:
        return None, (None, None), (None, None)  # no incident within time horizon

    #two solutions : once while approaching (entry) and once while moving apart (exit)
    #return the earlier among valid candidates (entry)
    t = min(t_candidates)
    
    #positions and distance at inicident
    d, p1, p2 = projected_separation(x1=x1, y1=y1, h1=h1, v1=v1, x2=x2, y2=y2, h2=h2, v2=v2, t=t)

    return t, p1, p2, d

#--------------------------------------------------------------------------

def find_nmac_ulm_ucm(au=None, ac=None, *,
                      xu:float=None, yu:float=None, hu:float=None, vu:float=None, 
                      xc:float=None, yc:float=None, hc:float=None, vc:float=None, wc:float=None,
                      nmac_distance:float=DEFAULT_NMAC_DIST, 
                      t_max:float=DEFAULT_MAX_EPISODE_TIME, 
                     ) -> tuple :      # -> Optional[Tuple[float, Tuple[float,float], Tuple[float,float]]]:
    """
    Compute the first time two aircraft come within nmac_distance, before t_max.
    Handles the case in which an airplane is in linear and the other in circular regular motion.

    Aircraft tuples: (x, y, heading, v) and (x, y, heading, v, omega)
    Returns: t_hit, p1_hit, p2_hit or None if no incident occurs within t_max.
    """
    #get data from airplane object if given
    if au is not None:
        xu, yu, hu, vu = get_airplane_ulm_data(au)
    if ac is not None:
        xc, yc, hc, vc, wc = get_airplane_data(ac)
    
    # -------------------------
    # Prune interval of search
    # -------------------------
    
    t_start, t_end = 0.0, t_max

    #circular motion airplane trajectory (center and radius)
    radius = vc/abs(wc)
    x = xc - radius*math.sin(hc)*math.copysign(1,wc)    #center x of circular trajectory
    y = yc + radius*math.cos(hc)*math.copysign(1,wc)    #center y of circular trajectory
    
    #linear motion airplane v components
    vux, vuy = math.cos(hu)*vu, math.sin(hu)*vu
    
    #quadradic equation in relation to the center of circular trajectory
    a = vux*vux + vuy*vuy
    b = 2*((xu-x)*vux + (yu-y)*vuy)
    c = (xu-x)*(xu-x) + (yu-y)*(yu-y) - (radius+nmac_distance)*(radius+nmac_distance)

    if a < TOL_ZERO:
        if math.sqrt(c) <= TOL_ZERO:
            #cannot filter    in this case, we have two static points ! ?
            t_start, t_end = 0.0, t_max
        else:
            #no solution, incident is impossible
            return None, (None, None, None), (None, None, None), None   #t, p1, p2, d
    else:
        d = b*b - 4*a*c
        if d < 0:
            #no real solution for the quadratic equation, no incident
            return None, (None, None, None), (None, None, None), None   #t, p1, p2, d
        sqrt_d = math.sqrt(d)
        t_low  = (-b - sqrt_d)/(2*a)
        t_high = (-b + sqrt_d)/(2*a)
        t_start = max(0.0, t_low)
        t_end = min(t_max, t_high)
        if t_start > t_end:
            return None, (None, None, None), (None, None, None), None   #t, p1, p2, d
        else:
            return find_nmac_by_sampling(x1=xu, y1=yu, h1=hu, v1=vu,
                                         x2=xc, y2=yc, h2=hc, v2=vc, w2=wc,
                                         nmac_distance=nmac_distance, t_start=t_start, t_end=t_end)
    
#--------------------------------------------------------------------------

def find_nmac_ucm_ucm(a1=None, a2=None, *,
                      x1:float=None, y1:float=None, h1:float=None, v1:float=None, w1:float=0.0,
                      x2:float=None, y2:float=None, h2:float=None, v2:float=None, w2:float=0.0,
                      nmac_distance:float=DEFAULT_NMAC_DIST, t_max:float=DEFAULT_MAX_EPISODE_TIME):      
                      # -> Optional[Tuple[float, Tuple[float,float], Tuple[float,float]]]:
    """
    Compute the first time two aircraft come within nmac_distance, before t_max.
    Handles the case in which both airplanes are in circular regular motion.

    Aircraft tuples: (x, y, heading, v, omega)
    Returns: t_hit, p1_hit, p2_hit or None if no incident occurs within t_max.
    """
    #get data from airplane object if given
    if a1 is not None:
        x1, y1, h1, v1, w1 = get_airplane_data(a1)
    if a2 is not None:
        x2, y2, h2, v2, w2 = get_airplane_data(a2)
    
    #prune interval:
    #(t_start, t_end) interval where planes can potentially come closer than nmac_distance. None if impossible.
    t_start, t_end = 0.0, t_max
    
    R1 = v1/abs(w1)
    R2 = v2/abs(w2)

    # Circle centers
    xc1 = x1 - R1*math.sin(h1)*math.copysign(1,w1)
    yc1 = y1 + R1*math.cos(h1)*math.copysign(1,w1)
    xc2 = x2 - R2*math.sin(h2)*math.copysign(1,w2)
    yc2 = y2 + R2*math.cos(h2)*math.copysign(1,w2)

    #distance of centers
    d_centers = math.hypot(xc2 - xc1, yc2 - yc1)
    #d_centers = math.dist((xc1,yc1),(xc2,yc2))

    """Check if two circles with centers c1, c2 and radii r1, r2
    can ever be within nmac_distance of each other."""
    #quick pruning: too far apart
    if d_centers > R1 + R2 + nmac_distance:
        print("circular tracks are too far away.")
        return None, (None, None, None), (None, None, None), None   #t, p1, p2, d
    #quick pruning: small circular trajectory inside big circular trajectory
    if d_centers < abs(R1 - R2) - nmac_distance:
        print("circular tracks are too far inside.")
        return None, (None, None, None), (None, None, None), None   #t, p1, p2, d
    
    t_start = 0.0
    t_end = t_max
    
    #allowed angular deviation using law of cosines
    #arg1 = (d_centers**2 + R1**2 - (R2+nmac_distance)**2)/(2*d_centers*R1)
    #arg2 = (d_centers**2 + R2**2 - (R1+nmac_distance)**2)/(2*d_centers*R2)
    #arg1 = max(-1.0, min(1.0,arg1))
    #arg2 = max(-1.0, min(1.0,arg2))
    #delta_theta1 = math.acos(arg1)
    #delta_theta2 = math.acos(arg2)
    #
    # Center angles
    #theta1_center = math.atan2(yc2 - yc1, xc2 - xc1) - h1
    #theta2_center = math.atan2(yc1 - yc2, xc1 - xc2) - h2
    #
    # Convert angular ranges to time intervals
    #t1_start = max(0.0, (theta1_center - delta_theta1)/w1)
    #t1_end   = min(t_max, (theta1_center + delta_theta1)/w1)
    #t2_start = max(0.0, (theta2_center - delta_theta2)/w2)
    #t2_end   = min(t_max, (theta2_center + delta_theta2)/w2)
    #
    #t_start = max(t1_start, t2_start)
    #t_end   = min(t1_end, t2_end)
    #
    #if t_start > t_end:
    #    print("angular range complete prunning.")
    #    return None, (None, None), (None, None)
        
    return find_nmac_by_sampling(x1=x1, y1=y1, h1=h1, v1=v1, w1=w1,
                                 x2=x2, y2=y2, h2=h2, v2=v2, w2=w2,
                                 nmac_distance=nmac_distance, t_start=t_start, t_end=t_end)

#--------------------------------------------------------------------------
    
def find_nmac_by_sampling(a1=None, a2=None, *,
                          x1:float=None, y1:float=None, h1:float=None, v1:float=None, w1:float=0.0,
                          x2:float=None, y2:float=None, h2:float=None, v2:float=None, w2:float=0.0,
                          nmac_distance:float=DEFAULT_NMAC_DIST, 
                          t_start:float=0.0, t_end:float=DEFAULT_MAX_EPISODE_TIME,
                          batch_samples:int=100,  #gives the number of uniform samples per batch
                         ):      # -> Optional[Tuple[float, Tuple[float,float], Tuple[float,float]]]:
    #get data from airplane object if given
    if a1 is not None:
        x1, y1, h1, v1, w1 = get_airplane_data(a1)
    if a2 is not None:
        x2, y2, h2, v2, w2 = get_airplane_data(a2)

    #time interval
    total_t = t_end-t_start
    #at least 3 samples
    batch_samples = max(3, batch_samples)
    #adjust sampling interval
    dt = total_t / (batch_samples-1)

    #iterate over samples until finding the first that violates constraint, otherwise get the sample with minimal distance, and search around
    while dt > TOL_TIME:

        #first time sample
        d_start, p1_start, p2_start = projected_separation(x1=x1, y1=y1, h1=h1, v1=v1, w1=w1, x2=x2, y2=y2, h2=h2, v2=v2, w2=w2, t=t_start)
    
        #initialize earliest point as the best nmac candidate
        d_nmac, p1_nmac, p2_nmac, t_nmac = d_start, p1_start, p2_start, t_start

        #violation from the beginning
        if d_nmac <= nmac_distance:
            #if first point distance violates nmac_distance constraint, stop searching and returns it
            return d_nmac, p1_nmac, p2_nmac, t_nmac
            
        #iterative uniform sampling
        else:

            #uniform sampling from earliest to latest sample
            for i in range(1, batch_samples):

                #get sampled time
                t = t_start + i*dt
                #relative distance at time t
                d, p1, p2 = projected_separation(x1=x1, y1=y1, h1=h1, v1=v1, w1=w1, x2=x2, y2=y2, h2=h2, v2=v2, w2=w2, t=t)
                
                #remember the best nmac candidate (closest of nmac violation) time found
                if d < d_nmac:
                    d_nmac, p1_nmac, p2_nmac, t_nmac = d, p1, p2, t

                #if distance violates nmac_distance constraint
                if d_nmac <= nmac_distance:

                    #nmac found, search around to increase precison
                    while dt > TOL_TIME:
                    
                        #t_start is before nmac
                        t_start = max(0.0, t_nmac-dt)
                        #t_end is after nmac
                        t_end = min(t_end, t_nmac)

                        #too small, stop search and return nmac approximation
                        interval = (t_end-t_start)
                        if interval < TOL_TIME:

                            return d_nmac, p1_nmac, p2_nmac, t_nmac

                        else:

                            #look in the middle
                            t = (t_start+t_end)/2
                            dt = dt/2
                            d, p1, p2 = projected_separation(x1=x1, y1=y1, h1=h1, v1=v1, w1=w1, x2=x2, y2=y2, h2=h2, v2=v2, w2=w2, t=t) 
                            #binary search
                            #if middle point is already in nmac
                            if d <= nmac_distance:
                                #it becomes t_end
                                t_end = t
                            #if middle is not yet in nmac
                            else:
                                #candidate becomes t_start
                                t_start = t

            #for loop ended without finding a constraint violation, affinate focus of search
            else:
                
                #get time before best candidate and time after best candidate
                t_start = max(0.0, t_nmac-dt)
                t_end = min(t_end, t_nmac+dt)
                dt = dt / 10.0

    #end while --> no violation
    return None, (None, None, None), (None, None, None), None   #t, p1, p2, d
            
#--------------------------------------------------------------------------

def find_nmac(a1=None, a2=None, *,
              x1:float=None, y1:float=None, h1:float=None, v1:float=None, w1:float=0.0,
              x2:float=None, y2:float=None, h2:float=None, v2:float=None, w2:float=0.0,
              nmac_distance:float=DEFAULT_NMAC_DIST, 
              t_max:float=DEFAULT_MAX_EPISODE_TIME
             ):      # -> Optional[Tuple[float, Tuple[float,float], Tuple[float,float]]]:
    """
    Compute the first time two aircraft come within nmac_distance, before t_max.
    Handles linear and circular regular motion (uniform constant velocities) for each airplane.

    Aircraft tuples: (x, y, heading, v, omega)
    omega=0 means linear motion
    Returns: t_hit, p1_hit, p2_hit or None if no incident occurs within t_max.
    """
    #get data from airplane object if given
    if a1 is not None:
        x1, y1, h1, v1, w1 = get_airplane_data(a1)
    if a2 is not None:
        x2, y2, h2, v2, w2 = get_airplane_data(a2)

    # --------------------------
    # Case 1: both aircraft in linear motion --> analytic quadratic exact solution
    # --------------------------
    if abs(w1) < TOL_ZERO and abs(w2) < TOL_ZERO:    #both omega are (near) zero (no angular velocity)
    
        print("ULM-ULM")
        return find_nmac_ulm_ulm(x1=x1, y1=y1, h1=h1, v1=v1, 
                                 x2=x2, y2=y2, h2=h2, v2=v2,
                                 nmac_distance=nmac_distance, t_max=t_max)

    # --------------------------
    # Case 2: one aircraft in linear motion and the other in circular motion
    # --------------------------

    elif abs(w1) < TOL_ZERO:
        
        #a1 is in ULM and a2 in UCM
        print("ULM-UCM")
        return find_nmac_ulm_ucm(xu=x1, yu=y1, hu=h1, vu=v1,  
                                 xc=x2, yc=y2, hc=h2, vc=v2, wc=w2,
                                 nmac_distance=nmac_distance, t_max=t_max)
    
    elif abs(w2) < TOL_ZERO:
    
        #a2 is in ULM and a1 in UCM
        print("UCM-ULM")
        return find_nmac_ulm_ucm(xu=x2, yu=y2, hu=h2, vu=v2,  
                                 xc=x1, yc=y1, hc=h1, vc=v1, wc=w1,
                                 nmac_distance=nmac_distance, t_max=t_max)
        
    # --------------------------
    # Case 3: both aircraft in circular motion
    # --------------------------

    else:

        #a2 is in ULM and a1 in UCM
        print("UCM-UCM")
        return find_nmac_ucm_ucm(x1=x1, y1=y1, h1=h1, v1=v1, w1=w1, 
                                 x2=x2, y2=y2, h2=h2, v2=v2, w2=w2,
                                 nmac_distance=nmac_distance, t_max=t_max)

    

###############################################################################


#if the parameter is a tuple, consider as (min, max) for random uniform sample
#if the parameter is a list, consider it for random uniform choice
#if the parameter is a value, simply return it
#if parameter is None, then use default

def pick_value(v, default=None, rng_or_seed=None):
    """
    This function serves to ransomization.
    Returns v if it is a single value (no randomization), or draws from a uniform distribution if v is an interval (tuple),
    or picks uniformly from a v if it is a list.
    If v is None, returns the default value.

    Args:
        v (None|int|float|tuple|list): the base value or list or interval.
        default (None|int|float|tuple|list): the default value to be used if v is None.
        rng_or_seed : a random_numpy_generator, or the seed to initialize one, or None.

    Returns:
        tuple(float, float, float): corresponding (x_t, y_t, heading_t).
    """
    
    #uses numpy random number generator - when seed is defined, it allows to reproduce sequence of random numbers.
    rng = np.random.default_rng(rng_or_seed)
    
    if v is None and default is not None:
        v = default
        
    if isinstance(v, tuple):    # and len(v)==2
        return rng.uniform(*v)
    elif isinstance(v, (list, range, np.ndarray)):
        return rng.choice(v)
    else:
        return v

###############################################################################

# class HorizontalAirplaneState():
    # """
    # Internally, the state of the Horizontal Airplane with (x, y, heading, v, omega) is represented using a NumPy array.

    # This class wraps a NumPy array to store the state elements. 
    # The default initialization is setting all to zero, or missing components in data with zeros. 
    # It supports arithmetic operations and NumPy functions (over state) seamlessly.
    
    # Supports single state (1D) or batch (2D or higher).
    
    # """

    # #additional_batch_shape:tuple=()

    # #position of each attribute in the state array
    # X_IDX = 0     #cartesian coordinate x (ft)
    # Y_IDX = 1     #cartesian coordinate y (ft)
    # HEAD_IDX = 2  #heading angle (yaw) (rad)
    # V_IDX = 3     #tangetial velocity (ft/s)
    # OMEGA_IDX = 4 #angular velocity (rad/s)

    # STATE_LEN = 5
    
    # def __init__(self):
    
        # data = np.asarray(data, dtype=float)
        # x, y, heading, v, omega = self.data[:5]
    
        # #state as np array [x, y, heading, v, omega]
        # self.state = np.zeros((self.STATE_LEN,), dtype=float)
    
    # # PROPERTIES
    
    # def get_x(self) -> float :
        # return self.state[self.X_IDX]
        # #return self[..., X_IDX]

    # def set_x(self, value:float):
        # self.state[self.X_IDX] = value    
        
    # x = property(get_x, set_x)

    # def __array__(self, dtype=None):
        # """Allow seamless use in NumPy functions."""
        # return np.asarray(self.state, dtype=dtype)        

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # """Allow seamless use with arithmetic operations."""
        # new_inputs = [np.asarray(i.state if isinstance(i, HorizontalAirplane) else i) for i in inputs]
        # result = getattr(ufunc, method)(*new_inputs, **kwargs)
        # if isinstance(result, np.ndarray) and result.shape == (self.STATE_LEN,):
            # return HorizontalAirplane(result)
        # return result

    # def __array_function__(self, func, types, args, kwargs):
        # """Allow seamless use with NumPy array function and reductions."""
        # result = func(*[np.asarray(a) if isinstance(a, HorizontalAirplane) else a for a in args], **kwargs)
        # if isinstance(result, np.ndarray) and result.shape == (self.STATE_LEN,):
            # return HorizontalAirplane(result)
        # return result
    

###############################################################################

# from array import array

# class Data(array):

    # def __new__(cls, data=None, *, a=None, b=None, array_dtype='f'):
        # if data is not None:
            # if isinstance(data, dict):
                # return array.__new__(cls, array_dtype, (data["a"], data["b"]))
            # elif isinstance(data, (list,tuple,np.ndarray)):
                # return array.__new__(cls, array_dtype, (data[:2]))
            # else:
                # return array.__new__(cls, array_dtype, (data.a, data.b))
        # else:
            # return array.__new__(cls, array_dtype, (a, b))
    
    # @property
    # def a(self):
        # return self[0]

    # @property
    # def b(self):
        # return self[1]
        
    # def get_data(self):
        # return self

    # def to_tuple(self):
        # return tuple(self)

    # def to_list(self):
        # return self.tolist()

    # def to_dict(self):
        # return {'a':self[0], 'b':self[1]} 
        
    # def to_numpy(self):
        # """Return as NumPy array view (zero-copy)."""
        # return np.frombuffer(self, dtype='f', count=2)

    # # ---- NumPy interoperability ----
    # def __array__(self, dtype=None, copy=False):
        # arr = np.frombuffer(self, dtype='f', count=2)
        # return arr.astype(dtype) if dtype else arr
        
    # #numpy compatibility protocol
    # def __array_interface__(self):
        # return {'shape':(2,), 'typestr':'<f4', 'version':3}

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # # Convert Data inputs to numpy arrays
        # np_inputs = [np.asarray(i) if isinstance(i, Data) else i for i in inputs]
        # result = getattr(ufunc, method)(*np_inputs, **kwargs)
        # # If result is a numpy array of shape (2,), wrap back into Vec2
        # if isinstance(result, np.ndarray) and result.shape == (2,):
            # return Data(result)
        # return result
        
    # # ---- Operators (forward to NumPy ufuncs) ----
    # def __add__(self, other): return np.add(self, other)
    # def __radd__(self, other): return np.add(other, self)
    # def __sub__(self, other): return np.subtract(self, other)
    # def __rsub__(self, other): return np.subtract(other, self)
    # def __mul__(self, other): return np.multiply(self, other)
    # def __rmul__(self, other): return np.multiply(other, self)
    # def __truediv__(self, other): return np.divide(self, other)
    # def __rtruediv__(self, other): return np.divide(other, self)
    
    # #attributes can be read as dictionary indices
    # def __getitem__(self, arg):
        # if type(arg) == int:
            # return super().__getitem__(arg)
        # else:
            # return self[('a','b').index(arg)]

    # def __str__(self):
        # return f"a={self[0]}, b={self[1]}"

    # def __len__(self):
        # return 2

###############################################################################

class HorizontalAirplane():
    """
    Class that implements a simplified Airplane for the Horizontal ACAS-Xu problem.
    The airplane has a x, y, heading position, tangential and angular velocities, but no vertical information (altitude, pitch...).
    There is no acceleration in the model, and changing angular or tangetial velocities is immediate and has imediate effect (Dubins model).
    
    For simuation purposes, the initialization supports control over randomization.
    Each attribute (x, y, heading, v, omega) can be initialized with exact values (no randomization) or intervals (tuple) or lists.
    In that case, the fisr initialization will randomize those values drawing from the interval or the list.
    However, the reset function will not randomize it again, but will come back to the same initial state.
    To randomize it again, there is a randomize function.
    It is useful to create randomized initial states, and reproduce it again in other simulations.

    Attributes:
        data (np.ndarray|None): Internal array of shape (5) storing x, y, heading, v, and omega.

    Properties:
        x: View of the x-coordinate
        y: View of the y-coordinate
        heading: View of the heading
        v: View of the linear (tangential) speed
        omega: View of omega
    """    
       
    def __init__(self, data=None, *,
                 x:float=None, y:float=None,  #position in cartesian coordinates  (ft)
                 heading:float=None,          #heading angle (rad)
                 v:float=None,                #tangential speed  (ft/s)
                 omega:float=None,            #angular speed  (rad/s)   #replacing last_a=None,
                 name:str='airplane',
                 rng_or_seed=None,          #numpy random generator, or initial seed for randomization
                ):
        #angles are given in radians.
        #time unit is generally second (s)
        #distance unit is generally feet (ft)
        #then, by convetion, distance in ft, speed in ft/s, head in rad, omega in rad/s
        #if tuple is given, then uniform random between interval
        """
        Initialize the airplane.
        Without any information, properties x, y, heading, and v are initialized with random values, except omega, which is initialized with zero.
        An airplace can be initialized with an array-like having 5 elements in this order: x, y, heading, v, omega. Or explicitly.
         
        
        Args:
            data (array_like): array of data with shape (5). 
            Missing components are padded with zeros. 
        
            x
            y
            heading
            v
            omega
        
        """

        #random generator
        self.rng = np.random.default_rng(rng_or_seed)
      
        #if data is given, it should corresponds to exact values
        if data is not None:
            x, y, heading, v, omega = get_airplane_data(data)

        #define initial state values for reset: they can be linearly randomized if an initial interval is given (function pick_value) 
        self.init_x =         pick_value(x,       default=(-DEFAULT_MAX_ACAS_DISTANCE, +DEFAULT_MAX_ACAS_DISTANCE), rng_or_seed=self.rng)     #cartesian positions to a central reference
        self.init_y =         pick_value(y,       default=(-DEFAULT_MAX_ACAS_DISTANCE, +DEFAULT_MAX_ACAS_DISTANCE), rng_or_seed=self.rng)
        self.init_heading =   pick_value(heading, default=(-math.pi, +math.pi), rng_or_seed=self.rng) #the heading is the yaw axis angle  (in rad)
        self.init_v =         pick_value(v,       default=(DEFAULT_MIN_SPEED, DEFAULT_MAX_SPEED), rng_or_seed=self.rng)  #speed in ft/s
        self.init_omega =     pick_value(omega,   default=0.0, rng_or_seed=self.rng)  #angular speed in rad/s
        
        #set airplane name
        self.name = name
                        
        #initialize state: (x, y, heading, v, omega)
        self.reset()

    #--------------------------------------------------------------------------
        
    #reset the value to the initialization values
    #if airplane was initialized using randomization, reset will recover the same values
    def reset(self, *,
              x:float=None, y:float=None, 
              heading:float=None, 
              v:float=None,
              omega:float=None,
              rng_or_seed=None):

        if rng_or_seed is None:
            rng_or_seed = self.rng
        
        #reset airplane state
        self.x =     pick_value(x,    default=self.init_x, rng_or_seed=rng_or_seed)          # x position (in ft, relative to some common referential)  
        self.y =     pick_value(y,    default=self.init_y, rng_or_seed=rng_or_seed)          # y position (in ft, relative to some common referential)  
        self.heading = pick_value(heading, default=self.init_heading, rng_or_seed=rng_or_seed)       # bearing, in rad
        self.v = pick_value(v, default=self.init_v, rng_or_seed=rng_or_seed)     # distance per unit of time (on heading direction) in ft/s
        self.omega = pick_value(omega, default=self.init_omega, rng_or_seed=rng_or_seed)   # angular speed
        
    #--------------------------------------------------------------------------

    #reset forcing randomization
    def randomize(self, *,
                  x=(-DEFAULT_MAX_ACAS_DISTANCE, +DEFAULT_MAX_ACAS_DISTANCE), y=(-DEFAULT_MAX_ACAS_DISTANCE, +DEFAULT_MAX_ACAS_DISTANCE), 
                  heading=(-math.pi, +math.pi),
                  v=(DEFAULT_MIN_SPEED, DEFAULT_MIN_SPEED),
                  omega=0.0,
                  rng_or_seed=None):  # distance in ft, speed in ft/s, angles in rad
    
        self.reset(x=x, y=y, heading=heading, v=v, omega=omega, rng_or_seed=rng_or_seed)    

    
    #--------------------------------------------------------------------------
    
    @property
    def heading_deg(self) -> float :
        return math.degrees(self.heading)

    @property
    def omega_deg(self) -> float :
        return math.degrees(self.omega)

    #--------------------------------------------------------------------------

    def coords(self) -> tuple :
        return (self.x, self.y)

    def pos(self) -> tuple :
        return (self.x, self.y, self.heading)
    
    #--------------------------------------------------------------------------

    def __repr__(self):
        return f"<HorizontalAirplane: shape={self.data.shape}, data={self.data}>"

    def __str__(self):
        return f"{self.name} : x = {round(self.x, 2)} {DISTANCE_UNIT_STR}, y = {round(self.y, 2)} {DISTANCE_UNIT_STR}, head = {round(self.heading_deg, 2)} rad, speed: {round(self.v, 2)} {DISTANCE_UNIT_STR}/{TIME_UNIT_STR}"

    #--------------------------------------------------------------------------

    #attributes can be read as dictionary indices
    def __getitem__(self, arg:str):
        return getattr(self, arg, None)
    
    #--------------------------------------------------------------------------

    def to_tuple(self):
        return (self.x, self.y, self.heading, self.v, self.omega)
        
    __iter__ = to_tuple

    def to_list(self):
        return [self.x, self.y, self.heading, self.v, self.omega]

    def to_dict(self):
        return {"x":self.x, "y":self.y, "heading":self.heading, "v":self.v, "omega":self.omega}

    def to_numpy(self):
        return np.array(self.to_list())

    #--------------------------------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        """Allow seamless use in NumPy functions."""
        return np.asarray(self.to_list(), dtype=dtype)

    def __len__(self):
        return 5

    #--------------------------------------------------------------------------

    def get_encounter_data(self, intruder=None, 
                           *, intruder_x:float=None, intruder_y:float=None, intruder_heading:float=None, intruder_v:float=None):
        
        if intruder is not None:
            intruder_x = intruder.x
            intruder_y = intruder.y
            intruder_heading = intruder.heading
            intruder_v = intruder.v
            
        #components of distance
        d_x = intruder_x - self.x
        d_y = intruder_y - self.y
        
        # relative distance  and  relative positional angle
        rho, theta_rad = polar_coords(d_x, d_y)
        #rho = math.dist( self.coords, intruder.coords)
        
        #heading difference
        psi_rad   = rad_mod(intruder_heading - self.heading)   
        
        return (rho, theta_rad, psi_rad, self.v, intruder_v)

    #--------------------------------------------------------------------------

    def projected_position(self, t:float=1.0):
        return projected_position(x=self.x, y=self.y, heading=self.heading, v=self.v, omega=self.omega, t=t)


    #--------------------------------------------------------------------------

    
###############################################################################
# OBSERVATION
#################

class HorizontalObservation():
            
    def __init__(self, data=None, *,
                 own=None, intruder=None,
                 omega=None, rho:float=None, theta:float=None, psi:float=None, v_own:float=None, v_int:float=None,  # distance in ft, speed in ft/s, angles in rad
                ):
        
        #get data from given airplanes
        if own is not None and intruder is not None:
            omega, rho, theta, psi, v_own, v_int = get_observation_data_from_airplanes(own=own, intruder=intruder)
        
        #get data from list, tuple, array, dixtionary or encounter-like object
        elif data is not None:
            omega, rho, theta, psi, v_own, v_int = get_observation_data(data)
        
        #attributes of an ACAS horizontal encounter
        self.omega  = omega   # distance in ft
        self.rho    = rho     # distance in ft
        self.theta  = theta   # relative angle where the intruder is in relation to own heading, in rad
        self.psi    = psi     # intruder heading, relative to own heading, in rad
        self.v_own  = v_own   # own tangential speed, in ft/s   
        self.v_int  = v_int   # intrueder tangential speed, in ft/s   
         
    #--------------------------------------------------------------------------
    
    #redefine values reusing the object
    update = __init__

    #--------------------------------------------------------------------------

    def __repr__(self):
        return f"<HorizontalObservation: shape=(5,), data={{omega={omega}, rho={rho}, theta={theta}, psi={psi}, v_own={v_own}, v_int={v_int} }}>"

    def __str__(self):
        return f"omega={round(self.omega, 2)}rad/{TIME_UNIT_STR}, rho={round(self.rho, 2)}{DISTANCE_UNIT_STR}, theta={round(self.theta, 2)} rad, psi={round(self.psi, 2)}rad, v_own={round(self.v_own, 2)}{DISTANCE_UNIT_STR}/{TIME_UNIT_STR}, , v_int={round(self.v_int, 2)}{DISTANCE_UNIT_STR}/{TIME_UNIT_STR}"

    #--------------------------------------------------------------------------

    #attributes can be read as dictionary indices
    def __getitem__(self, arg:str):
        return getattr(self, arg, None)
    
    #--------------------------------------------------------------------------

    def to_tuple(self):
        return (self.x, self.y, self.heading, self.v, self.omega)
        
    __iter__ = to_tuple

    def to_list(self):
        return [self.x, self.y, self.heading, self.v, self.omega]

    def to_dict(self):
        return {"x":self.x, "y":self.y, "heading":self.heading, "v":self.v, "omega":self.omega}

    def to_numpy(self):
        return np.array(self.to_list())

    #--------------------------------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        """Allow seamless use in NumPy functions."""
        return np.asarray(self.to_list(), dtype=dtype)

    def __len__(self):
        return 5

    #--------------------------------------------------------------------------   

    
###############################################################################
# ENCOUNTER
####################

class HorizontalEncounter():

    def __init__(self, enc=None, *,
                 own=None, intruder=None,
                 rho:float=None, theta:float=None, psi:float=None, v_own:float=None, v_int:float=None,  # distance in ft, speed in ft/s, angles in rad
                ):
        
        #get data from given airplanes
        if own is not None and intruder is not None:
            rho, theta, psi, v_own, v_int = get_encounter_data_from_airplanes(own=own, intruder=intruder)
        
        #get data from list, tuple, array, dixtionary or encounter-like object
        elif enc is not None:
            rho, theta, psi, v_own, v_int = get_encounter_data(enc)
           
        #set attributes (rho, theta, psi, v_own, v_int), and also (rho_x, rho_y).
        self.rho    = rho     # distance in ft
        self.theta  = theta   # relative angle where the intruder is in relation to own heading, in rad
        self.psi    = psi     # intruder heading, relative to own heading, in rad
        self.v_own  = v_own   # own tangential speed, in ft/s   
        self.v_int  = v_int   # intrueder tangential speed, in ft/s   

        #components x and y of rho, correponding to x_intruder and y_intruder when own x, y, h are all zero. 
        #note that psi corresponds to h_intruder in this same case.
        self.rho_x, self.rho_y = cartesian_coords(self.rho, self.theta)

    #--------------------------------------------------------------------------
    
    #redefine values reusing the object
    update = __init__

    #--------------------------------------------------------------------------

    def create_airplanes(self):
        own = HorizontalAirplane(x=0.0, y=0.0, heading=0.0, v=self.v_own, omega=OMEGA_COC, name="own")
        intruder = HorizontalAirplane(x=self.rho_x, y=self.rho_y, heading=self.psi, v=self.v_int, omega=OMEGA_COC, name="intruder")
        return own, intruder

    #--------------------------------------------------------------------------

    def projected_separation(self, t):
        """
        Returns the projected distance to the intruder at a future time t, considering both aircraft in ULM movement.
        """
        return projected_separation(x1=0.0, y1=0.0, h1=0.0, v1=self.v_own, x2=self.rho_x, y2=self.rho_y, h2=self.psi, v2=self.v_int, t=t)
                
    #--------------------------------------------------------------------------

    def cpa(self):
        """
        Compute Closest Point of Approach (CPA) between own and intruder,
        supposing that both are flying in straight lines (ULM) at constant velocity (uniform linear motion).

        Note that:
            (x1, y1, h1, v1) is the own situation (x, y coordinates, heading angle in radians, speed) corresponding to (0, 0, 0, v_own)
            (x2, y2, h2, v2) is the intruder situation, corresponding to (rho_x, rho_y, psi, v_int).

        Returns:
            t_cpa (float): time to CPA (>=0)
            p1_cpa (tuple): position and heading of aircraft 1 at CPA
            p2_cpa (tuple): position and heading of aircraft 2 at CPA
            d_cpa (float): minimal separation distance
        """
    
        return find_cpa_ulm_ulm(x1=0.0, y1=0.0, h1=0.0, v1=self.v_own, 
                                x2=self.rho_x, y2=self.rho_y, h2=self.psi, v2=self.v_int)
        
   #--------------------------------------------------------------------------

    def nmac(self, nmac_distance:float=DEFAULT_NMAC_DIST, t_max:float=DEFAULT_MAX_EPISODE_TIME):
        """
        Find next time when the horizontal separation of two aircraft becomes less or equal than nmac_distance (nmac_distance).
        The airplanes are supposed to fly in uniform linear straight-motion (ULM).

        Args:
            nmac_distance: threshold or incident distance (positive).
            t_max: maximum time horizon to consider. None for infinite (ubounded horizon).
            
        Returns:
            (t, (x1_t, y1_t, h1_t), (x2_t, y2_t, h2_t), d) where
                t (float): incident time >= 0, when distance first becomes <= nmac_distance, or epsilon)
                (x1_t, y1_t, h1_t), (x2_t, y2_t, h2,t) (tuples): positions (x,y) and heading of aircraft 1 and 2 at incident time t
                d (float) distance at incident time.
            or tuples of None if no incident occurs in the interval [0, t_max].

        Notes:
            - If they are already closer than nmac_distance, violating the constraint at the initial time, the function returns t=0 and initial positions and distance.
        """
    
        return find_nmac_ulm_ulm(x1=0.0, y1=0.0, h1=0.0, v1=self.v_own, 
                                 x2=self.rho_x, y2=self.rho_y, h2=self.psi, v2=self.v_int, 
                                 nmac_distance=nmac_distance, 
                                 t_max=t_max)
        
   #--------------------------------------------------------------------------
            
    def printf(self):

        print(f"rho (distance, in ft) = {round(self.rho, 2)}")
        print(f"theta (relative intruder angle) = {round(self.theta, 2)} rad , = {round(self.theta/np.pi, 2)} pi rad , = {round(np.degrees(self.theta), 2)} deg")
        print(f"psi (relative intruder heading) = {round(self.psi, 2)} rad , = {round(self.psi/np.pi, 2)} pi rad , = {round(np.degrees(self.psi), 2)} deg")
        print(f"v_own (own speed, in ft/s) = {round(self.v_own, 2)}")
        print(f"v_int (intruder speed, in ft/s) = {round(self.v_int, 2)}")

    #--------------------------------------------------------------------------

    def to_list(self):
        return [self.rho, self.theta, self.psi, self.v_own, self.v_int]
    
    def to_tuple(self):
        return (self.rho, self.theta, self.psi, self.v_own, self.v_int)
    
    def to_dict(self):
        return {'rho':self.rho, 'theta':self.theta, 'psi':self.psi, 'v_own':self.v_own, 'v_int':self.v_int}

    #--------------------------------------------------------------------------

    #attributes can be read as dictionary indices
    def __getitem__(self, arg:str):
        if arg in ['rho', 'theta', 'psi', 'v_own', 'v_int', 'rho_x', 'rho_y']:
            return getattr(self, arg, None)

    def __str__(self):
        return f"{self.rho}: {round(self.rho, 2)} ft, theta: {round(self.theta, 2)} rad, psi: {round(self.psi, 2)} rad, own speed: {round(self.v_own, 2)} ft/s, intruder speed: {round(self.v_int, 2)} ft/s"

    #numpy compatibility protocol
    def __array_interface__(self):
        return {'shape':(5), 'typestr':'>f4', 'version':3}

    #numpy compatibility protocol
    __array__ = to_tuple

    #iterator over attributes
    __iter__ = to_tuple

###############################################################################

def ensure_horizontal_encounter(enc=None, *, rho:float=None, theta:float=None, psi:float=None, v_own:float=None, v_int:float=None):
    if isinstance(enc, HorizontalEncounter):
        return enc
    else:
        return HorizontalEncounter(enc, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)

###############################################################################

def create_random_intruder(own, *, incident_time=(40,60), incident_distance=(0.0, DEFAULT_NMAC_DIST), rng_or_seed=None, verbose=False):
    
    rng = np.random.default_rng(rng_or_seed)
    
    incident_time = pick_value(incident_time, rng_or_seed=rng_or_seed)
    incident_distance = pick_value(incident_distance, rng_or_seed=rng_or_seed)
       
    #calculate own position at incident time
    own_x_incident = own.x + (own.v * incident_time * math.cos(own.heading))
    own_y_incident = own.y + (own.v * incident_time * math.sin(own.heading))

    #calculate intruder position at incident time
    theta = rng.uniform(-math.pi, +math.pi) #intruder relative angle at incident_time
    intruder_x_incident = own_x_incident + incident_distance * math.cos(theta)
    intruder_y_incident = own_y_incident + incident_distance * math.sin(theta)
 
    #create intruder
    intruder_head = rng.uniform(-math.pi, +math.pi)
    intruder_speed = rng.uniform(DEFAULT_MIN_SPEED, DEFAULT_MAX_SPEED)
    intruder_x = intruder_x_incident - (intruder_speed * incident_time * math.cos(intruder_head))
    intruder_y = intruder_y_incident - (intruder_speed * incident_time * math.sin(intruder_head))
    intruder = HorizontalAirplane(x=intruder_x, y=intruder_y, heading=intruder_head, v=intruder_speed, name="intruder")

    if verbose:
        print('incident time', incident_time)
        print('incident distance', incident_distance)
        print('incident own (x,y)', own_x_incident, own_y_incident)
        print('incident intruder (x,y)', intruder_x_incident, intruder_y_incident)

    return intruder
    
###############################################################################

def create_random_incident(incident_distance=(0.0, DEFAULT_NMAC_DIST), incident_time=(40,60), rng_or_seed=None, verbose=False):
    own = HorizontalAirplane(x=0., y=0., heading=0., name="own")  #random speed
    intruder = create_random_intruder(own, incident_distance=incident_distance, incident_time=incident_time, rng_or_seed=rng_or_seed, verbose=verbose)
    return  own, intruder

###############################################################################
    

if __name__ == '__main__' :

   from matplotlib import pyplot as plt
   
   print()
   print("TESTING MODULE FUNCTIONS...") 
   print('---------------------------')

   print()
   print("FUNCTION omega_to_act()")
   for name, w in zip(ACT_NAMES, ACT_OMEGA):
      print(name, w, ACT_NAMES[omega_to_act(w)])
   for name, w in zip(ACT_NAMES, ACT_OMEGA):
      print(name, w, ACT_NAMES[omega_to_act(w+0.0001)])

   print()
   print("FUNCTION projected_position()")
   print("Same airplane with different omega (angular velocity) : plot trajectories.") 

   x, y, h, v = .0, .0, HEADING_E, 1000.0
   trajs = [[projected_position(x=x, y=y, heading=h, v=v, omega=w, t=t) for t in range(20)] for w in ACT_OMEGA]
   fig, ax = plt.subplots()
   for traj, name in zip(trajs, ACT_NAMES):
      ax.plot([e[0] for e in traj], [e[1] for e in traj], 'o', label=name)
   ax.set_title(f"Same airplane with different $\omega$ (angular velocity)\n ({x},{y}), $\phi$={h}, v={v}")
   ax.legend()
   plt.show()

   print('-----------------')

   print()
   print("CLASS HorizontalAirplane")
   print("Creating Airplane using defaults") 

   print()
   a = HorizontalAirplane()
   print("init: ", a)
   a.reset()
   print("reset:", a)
   a.randomize()
   print("rand: ", a)
   a.reset()
   print("reset:", a)

   print()
   print("Creating Airplane using specific values") 

   print()
   a = HorizontalAirplane(x=100.0, y=200.0, heading=0.0, name='own')
   print("init: ", a)
   print(a.x, a.y, a.heading)
   a.reset()
   print("reset:", a)
   a.randomize()
   print("rand: ", a)
   a.reset()
   print("reset:", a)

   a = HorizontalAirplane([100.0, 200.0, 0.0, 500.0, 0.0])
   
   print()
   print('-----------------')
   
   print()
   print("CLASS HorizontalObservation")
   print("Creating Observations") 
   
   omega = OMEGA_COC   # 0.0 --> equiv to last_a = 0 #coc
   rho, theta, psi, v_own, v_int = 10000., -1., +1., 800., 750.
   
   obs = HorizontalObservation(omega=omega, rho=rho, theta=theta, psi=psi, v_own=v_own, v_int=v_int)
   print(obs)
   print('-----------------')
   obs = HorizontalObservation([omega, rho, theta, psi, v_own, v_int])
   print(obs)
   print('-----------------')

   obs = HorizontalObservation([omega, rho, theta, psi, v_own, v_int])
   print(obs)
   print('-----------------')
   
   
   print("Creating Encounter given inputs") 
   
   print()
   #simple test
   #last_a = 0 #coc
   rho, theta, psi, v_own, v_int = 10000, -1., +1, 800, 750
   #rho, theta, psi, v_own, v_int = 499, -3.1416, -3.1416, 50., 0.      
   #state = [last_command, rho, theta, psi, v_own, v_int]
   #state = [last_command, v_int, v_own, psi, theta, rho]
   #dic_state = {name:v for name, v in zip(lut.names, state)}

   print()
   print('-----------------')

   print("CLASS HorizontalEncounter")

   enc = HorizontalEncounter(v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)
   print('encounter: ', enc)
   
   intr_x, intr_y = cartesian_coords(rho, theta)
   print('intruder: x=', intr_x, "y=", intr_y)

   print()
   #print("Encounter:", enc)
   #print("distance (Km)", rho / 3281)
   #print("own speed (Km/s)", round(v_own / 3281, 2))
   #print("intruder speed (Km/s)", round(v_int / 3281, 2))
   #print("initial relative angle (degrees)", round(theta * 180 / math.pi, 2))
   #print("intruder cap angle (degrees)", round(psi * 180 / math.pi, 2))
   enc.printf()
   
   print()
   print("distance (in ft) t=0s:", enc.projected_separation(0))
   print("distance (in ft) t=1s:", enc.projected_separation(1))
   t_cpa, p_own_cpa, p_int_cpa, d_cpa = enc.cpa()
   print("minimal distance time:", t_cpa)
   print("minimal distance (in ft):", enc.projected_separation(t_cpa))
   t_nmac, p_own_nmac, p_int_nmac, d_nmac = enc.nmac()
   print("incident time:", t_nmac)
   print("incident distance:", d_nmac)

   print("CALC AGAIN")
   own, intruder = enc.create_airplanes()
   t_cpa, pos_own_cpa, pos_int_cpa, d = find_cpa_ulm_ulm(own, intruder)
   print("minimal distance time:", t_cpa)
   print("minimal distance (in ft):", d)
   t_nmac, pos_own_nmac, pos_int_nmac, d_nmac = find_nmac(own, intruder)
   print("incident time:", t_nmac)
   print("incident distance:", d_nmac)

   t_max = t_cpa or 100
      
   x, y, h, v = .0, .0, .0, v_own
   traj_own = [projected_position(x=x, y=y, heading=h, v=v, t=t) for t in range(int(t_max)+1)]
   traj_int = [projected_position(x=intr_x, y=intr_y, heading=psi, v=v_int, t=t) for t in range(int(t_max)+1)]
   fig, ax = plt.subplots()
   ax.plot([e[0] for e in traj_own], [e[1] for e in traj_own], 'o g', label='own')
   ax.plot([e[0] for e in traj_int], [e[1] for e in traj_int], 'o b', label='intruder')
   ax.plot(pos_own_nmac[0], pos_own_nmac[1], 'X g', label='own nmac')
   ax.plot(pos_int_nmac[0], pos_int_nmac[1], 'X b', label='intruder nmac')
   ax.plot(pos_own_cpa[0], pos_own_cpa[1], '* g', label='own cpa')
   ax.plot(pos_int_cpa[0], pos_int_cpa[1], '* b', label='intruder cpa')
   ax.set_title(f"HorizontalEncounter (ULM)\n$x_1={x}$, $y_1={y}$, $\\phi_1={h}$, $v_1={v}$\n $x_2={intr_x}$, $y_2={intr_y}$, $\\phi_2={psi}$, $v_2={v_int}$\n$\\rho={rho}$, $\\theta={theta}$, $\\psi={psi}$")
   ax.legend()
   plt.show()

   print()
   print("Creating Encounter given airplanes") 
   print()
   print("Creating Airplanes") 
   own = HorizontalAirplane(x=0.0, y=0.0, heading=0.0, v=v_own, name="own")
   intruder = HorizontalAirplane(x=intr_x, y=intr_y, heading=psi, v=v_int, name="intruder")   #random
   print(own)
   print(intruder)
   
   enc = HorizontalEncounter(own=own, intruder=intruder)

   print()
   #print("Encounter:", enc)
   enc.printf()
   print()
   print("distance (in ft) t=0s:", enc.projected_separation(0))
   print("distance (in ft) t=1s:", enc.projected_separation(1))
   t_cpa, pos_own_cpa, pos_int_cpa, d = enc.cpa()   
   print("minimal distance time:", t_cpa)
   print("minimal distance (in ft):", d_cpa)
   t_nmac, pos_own_nmac, pos_int_nmac, d_nmac = enc.nmac()
   print("incident distance time:", t_nmac)
   print("incident distance (in ft):", d_nmac)

   print()
   print('-----------------')
   print("Creating Encounter as a random incident") 
   print()
   own, intruder = create_random_incident()
   enc = HorizontalEncounter(own=own, intruder=intruder)
   print(own)
   print(intruder)
   #print("Encounter:", enc)
   print()
   enc.printf()
   print()
   print("distance (in ft) t=0s:", enc.projected_separation(0))
   print("distance (in ft) t=1s:", enc.projected_separation(1))
   t_cpa, pos_own_cpa, pos_int_cpa, d = find_cpa_ulm_ulm(own, intruder)
   print("minimal distance time:", t_cpa)
   print("minimal distance (in ft):", d)
   t_nmac, pos_own_nmac, pos_int_nmac, d_nmac = find_nmac(own, intruder)
   print("incident distance time:", t_nmac)
   print("incident distance (in ft):", d_nmac)

   t_max = t_cpa or 100
      
   traj_own = [projected_position(x=own.x, y=own.y, heading=own.heading, v=own.v, t=t) for t in range(int(t_max)+1)]
   traj_int = [projected_position(x=intruder.x, y=intruder.y, heading=intruder.heading, v=intruder.v, t=t) for t in range(int(t_max)+1)]
   fig, ax = plt.subplots()
   ax.plot([e[0] for e in traj_own], [e[1] for e in traj_own], 'o g', label="airplane 1")
   ax.plot([e[0] for e in traj_int], [e[1] for e in traj_int], 'o b', label="airplane 2")
   ax.plot(pos_own_nmac[0], pos_own_nmac[1], 'X g', label="nmac 1")
   ax.plot(pos_int_nmac[0], pos_int_nmac[1], 'X b', label="nmac 2")
   ax.plot(pos_own_cpa[0], pos_own_cpa[1], '* g', label="cpa 1")
   ax.plot(pos_int_cpa[0], pos_int_cpa[1], '* b', label="cpa 2")
   ax.set_title(f"Two airplanes (ULM)\n$x_1={own.x}$, $y_1={own.y}$, $\\phi_1={own.heading}$, $v_1={own.v}$\n $x_2={intruder.x}$, $y_2={intruder.y}$, $\\phi_2={intruder.heading}$, $v_2={intruder.v}$")
   ax.legend()
   plt.show()

   print()
   print("Creating non-linear Encounter") 
   print()
   print("Creating Airplanes") 
   own = HorizontalAirplane(x=-2000.0, y=0.0, heading=HEADING_S, v=1000.0, omega=OMEGA_WL, name="own")
   intruder = HorizontalAirplane(x=+2000.0, y=0.0, heading=HEADING_S, v=1000.0, omega=OMEGA_WR, name="intruder")   #random
   print(own)
   print(intruder)
   
   enc = HorizontalEncounter(own=own, intruder=intruder)

   print()
   #print("Encounter:", enc)
   enc.printf()
   print()
   t_cpa = None
   #t_cpa, pos_own_cpa, pos_int_cpa, d = find_cpa(own, intruder)
   #print("minimal distance time:", t_cpa)
   #print("minimal distance (in ft):", d)
   t_nmac, pos_own_nmac, pos_int_nmac, d_nmac = find_nmac(own, intruder)
   print("incident distance time:", t_nmac)
   print("incident distance (in ft):", d_nmac)

   t_max = t_cpa or 100
      
   traj_own = [projected_position(x=own.x, y=own.y, heading=own.heading, v=own.v, omega=own.omega, t=t) for t in range(1, int(t_max)+1)]
   traj_int = [projected_position(x=intruder.x, y=intruder.y, heading=intruder.heading, v=intruder.v, omega=intruder.omega, t=t) for t in range(1, int(t_max)+1)]
   fig, ax = plt.subplots()
   ax.plot(own.x, own.y, 's g', label="own start")
   ax.plot(intruder.x, intruder.y, 's b', label="intruder start")
   ax.plot([e[0] for e in traj_own], [e[1] for e in traj_own], 'o g', label="own")
   ax.plot([e[0] for e in traj_int], [e[1] for e in traj_int], 'o b', label="intruder")
   if t_nmac is not None:
      ax.plot(pos_own_nmac[0], pos_own_nmac[1], 'X', label="own nmac")
      ax.plot(pos_int_nmac[0], pos_int_nmac[1], 'X', label="intruder nmac")
   #ax.plot(pos_own_cpa[0], pos_own_cpa[1], '*')
   #ax.plot(pos_int_cpa[0], pos_int_cpa[1], '*')
   ax.set_title(f"Two airplanes (UCM)\n$x_1={own.x}$, $y_1={own.y}$, $\\phi_1={own.heading}$, $v_1={own.v}$, $\\omega_1={own.omega}$\n $x_2={intruder.x}$, $y_2={intruder.y}$, $\\phi_2={intruder.heading}$, $v_2={intruder.v}, $\\omega_2={intruder.omega}$")
   ax.legend()
   plt.show()
