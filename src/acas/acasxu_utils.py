# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

###############################################################################

import math
import numpy as np

from itertools import product as iterprod

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


from acas.acasxu_basics import ACT_NAMES, HorizontalObservation
from acas.acasxu_model_onnx_dubins import DubinsNN

###############################################################################

def plot_best_actions_cartesian(psi, v_own, v_int, last_a=0, max_dist=20000, dist_incr=1000, model=None, fig=None, show=True):

    if fig is None:    
        fig, ax = plt.subplots(figsize=(8,8))
    else:
        ax = fig.gca()
    
    #own displacement in ft in 10 seconds
    d_own = v_own * 10 
    
    #int displacement in ft in 10 seconds
    d_int = v_int * 10 
    
    #intruder displacement
    # psi = 0 means intruder flying in the same sense than own
    dx = d_int * np.cos(psi)
    dy = d_int * np.sin(psi)
    
    # draw own speed
    ax.annotate('', xytext=(0,0), xy=(d_own,0), arrowprops=dict(arrowstyle='-|>'))

    # draw intruder speed
    ax.annotate('', xytext=(-dx,-dy), xy=(0,0), arrowprops=dict(arrowstyle='-|>', linestyle='--', alpha=0.3))
    
    
    colors = ['whitesmoke', 'lightgreen', 'lightblue', 'darkgreen', 'darkblue']
    
    if model is None:
        model = DubinsNN()

    x_values = range(-max_dist, +max_dist, dist_incr)
    y_values = range(-max_dist, +max_dist, dist_incr)
    coords_x_y = np.array(list(iterprod(x_values, y_values)))
    rho_theta = np.array([(np.hypot(x, y), np.arctan2(y, x)) for (x, y) in coords_x_y])
    A = np.array([model.predict(HorizontalObservation(last_a=last_a, v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)).argmin() for (rho, theta) in rho_theta])
    C = np.array([colors[a] for a in A])    
    
#    #intruder position
#    for x in range(-max_dist, +max_dist, dist_incr):
#        for y in range(-max_dist, +max_dist, dist_incr):
#            
#            #rho = np.sqrt(x**2 + y**2)
#            rho = np.hypot(x, y)
#            theta = np.arctan2(y, x)
#            
#            action = model.predict(last_cmd, v_own, v_int, theta, psi, rho).argmin()
#    
#            #if action != 0:
#            # draw intruder position
#            ax.scatter(x, y, color=colors[action])  #, label=names[action])
#
#            # draw intruder speed
#            #if theta == 0:
#            #    ax.annotate('', xytext=(x,y), xy=(x+dx,y+dy), arrowprops=dict(arrowstyle='->', alpha=0.2))
    X = np.array(coords_x_y)[:,0]
    Y = np.array(coords_x_y)[:,1]
    ax.scatter(X, Y, color=C)  #, label=names[action])

    custom = [Line2D([], [], marker='.', markersize=20, c=color, linestyle='None') for color in colors]
    ax.legend(custom, ACT_NAMES, loc='upper left')
    
    if show:
        plt.show()
            
    
###############################################################################


def plot_best_actions(psi, v_own, v_int, last_a=0, model=None, fig=None, ax=None, show=True):

    if ax is None:
        if fig is None:    
            fig, ax = plt.subplots(figsize=(8,8))
        else:
            ax = fig.gca()
    
    #own displacement in ft in 10 seconds
    d_own = v_own*10  #in m : /3.281
    
    #int displacement in ft in 10 seconds
    d_int = v_int*10
    
    #intruder displacement
    # psi = 0 means intruder flying in the same sense than own
    dx = d_int * np.cos(psi)
    dy = d_int * np.sin(psi)
    
    # draw own speed
    ax.annotate('', xytext=(0,0), xy=(d_own,0), arrowprops=dict(arrowstyle='-|>'))
    
    # draw intruder speed
    ax.annotate('', xytext=(-dx,-dy), xy=(0,0), arrowprops=dict(arrowstyle='-|>', linestyle='--', alpha=0.3))
    
    #41 angles (rad) from -pi to +pi, linearly disposed
    #theta_values = [-3.1416, -2.9845, -2.8274, -2.6704, -2.5133, -2.3562, -2.1991, -2.042, -1.885, -1.7279, -1.5708, -1.4137, -1.2566, -1.0996, -0.9425, -0.7854, -0.6283, -0.4712, -0.3142, -0.1571, 0.0, 0.1571, 0.3142, 0.4712, 0.6283, 0.7854, 0.9425, 1.0996, 1.2566, 1.4137, 1.5708, 1.7279, 1.885, 2.042, 2.1991, 2.3562, 2.5133, 2.6704, 2.8274, 2.9845, 3.1416]
    theta_values = np.linspace(-3.1416, 3.1416, 41)   #rounded at 4 dec positions
    
    #39 distances
    rho_values = [   499.,    800.,   2508.,   4516.,   6525.,   8534.,  10543.,  12551.,  14560.,  
      16569.,  18577.,  20586.,  22595.,  24603.,  26612.,  28621.,  30630.,  32638.,
      34647.,  36656.,  38664., 40673.,  42682.,  44690.,  46699. , 48708.,  50717.,
      52725.,  54734.,  56743.,  58751.,  60760.,  75950.,  94178., 112406., 130634.,
     148862., 167090., 185318.]
    
    colors = ['whitesmoke', 'lightgreen', 'lightblue', 'darkgreen', 'darkblue']
    names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

    if model is None:
        model = DubinsNN()
    
    A = np.array([[model.predict(HorizontalObservation(last_a=last_a, v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)).argmin() for rho in rho_values] for theta in theta_values]).flat
    X = np.array([[rho * math.cos(theta) for rho in rho_values] for theta in theta_values]).flat
    Y = np.array([[rho * math.sin(theta) for rho in rho_values] for theta in theta_values]).flat
    C = np.array([colors[a] for a in A])    
    
#    for rho in rho_values:
#        for theta in theta_values:
#            #intruder position
#            x = rho * math.cos(theta)
#            y = rho * math.sin(theta)
#    
#            action = model.predict(last_cmd, v_own, v_int, theta, psi, rho).argmin()
#    
#            #if action != 0:
#            # draw intruder position
#            ax.scatter(x, y, color=colors[action])  #, label=names[action])
#
#            # draw intruder speed
#            #if theta == 0:
#            #    ax.annotate('', xytext=(x,y), xy=(x+dx,y+dy), arrowprops=dict(arrowstyle='->', alpha=0.2))

    ax.scatter(X, Y, color=C)  #, label=names[action])

    custom = [Line2D([], [], marker='.', markersize=20, c=color, linestyle='None') for color in colors]
    ax.legend(custom, names, loc='upper left')
    
    
    if show:
        plt.show()
            
    
###############################################################################


def plot_state(encounter, fig=None, animate=False, frames=12, show=True):

    #own displacement in ft in 10 seconds
    d_own = encounter.v_own * 10
    
    #int displacement in ft in 10 seconds
    d_int = encounter.v_int * 10
    
    #intruder position
    x = encounter.rho * math.cos(encounter.theta)
    y = encounter.rho * math.sin(encounter.theta)
    
    #intruder displacement
    # psi = 0 means intruder flying in the same sense than own
    dx = d_int * np.cos(encounter.psi)
    dy = d_int * np.sin(encounter.psi)

    #lim = 2.3 * max(x, y)

    if fig is None:    
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    
    #ax.set_xlim(-lim, lim)
    #ax.set_ylim(-lim, lim)

    # draw distance
    ax.annotate('', xytext=(x,y), xy=(0,0), arrowprops=dict(arrowstyle="-", linestyle="--", shrinkA=0, shrinkB=0, alpha=0.5))
    ax.annotate(f'$\\rho = {round(encounter.rho/1000,1)}$ Km', xy=(x//2 + encounter.v_own, y//2), xycoords='data')
    
    # draw own speed
    ax.annotate('', xytext=(0,0), xy=(d_own,0), arrowprops=dict(arrowstyle='->'))
    ax.text(encounter.v_own, -2*encounter.v_own, str(round(d_own/1000,1)) + "Km in 10s")
    # draw own linear trajectory
    ax.annotate('', xy=(12*encounter.v_own,0), xytext=(0,0), arrowprops=dict(arrowstyle="-", linestyle=":", color='b', shrinkA=0, shrinkB=0, alpha=0.5))
    # draw intruder speed
    ax.annotate('', xytext=(x,y), xy=(x+dx,y+dy), arrowprops=dict(arrowstyle='->'))
    # draw own position
    ax.scatter(0, 0)
    ax.scatter(d_own, 0, alpha=0)
    # draw intruder position
    ax.scatter(x, y)
    ax.scatter(x+dx, y+dy, alpha=0)
    
    # draw intruder angle
    r = encounter.v_int
    arc_angles = np.linspace(0, encounter.psi, 20)
    arc_xs = r * np.cos(arc_angles)
    arc_ys = r * np.sin(arc_angles)
    ax.plot(arc_xs+x, arc_ys+y, lw = 1, color='k', alpha=0.5)
    ax.annotate(f'$\\psi = {round(np.degrees(encounter.psi),1)}\\degree$', xy=(x+encounter.v_int, y+encounter.v_int//2), xycoords='data')
    
    # draw intruder reference own heading
    ax.annotate('', xy=(x+2*encounter.v_int,y), xytext=(x,y), arrowprops=dict(arrowstyle="-", linestyle=":", color='g', shrinkA=0, shrinkB=0, alpha=0.5))

    # draw theta angle and annotation
    r = encounter.v_own
    arc_angles = np.linspace(0, encounter.theta, 20)
    arc_xs = r * np.cos(arc_angles)
    arc_ys = r * np.sin(arc_angles)
    ax.plot(arc_xs, arc_ys, lw = 1, color='k', alpha=0.5)
    ax.annotate(f'$\\theta = {round(np.degrees(encounter.theta),1)}\\degree$', xy=(encounter.v_own, encounter.v_own//2), xycoords='data')
    #plt.gca().annotate('<----- r = 1.5 ---->', xy=(0 - 0.2, 0 + 0.2), xycoords='data', fontsize=15, rotation = 45)
    
    ax.set_aspect('equal')
    
    ax.set_title("displacement (interval = $10s$)")
    
    if animate:
       # draw own final position
       ax.scatter(frames*d_own, 0)
       # draw intruder final position
       ax.scatter(x+frames*dx, y+frames*dy)
    
       def animate(n):
          own_traj_coordinates = np.array([(0,0), ((n+1)*d_own, 0)])
          own_lines, = ax.plot(own_traj_coordinates[:,0], own_traj_coordinates[:,1], alpha=0.4, color='b')
          intruder_traj_coordinates = np.array([(x,y), (x+(n+1)*dx, y+(n+1)*dy)])
          intruder_lines, = ax.plot(intruder_traj_coordinates[:,0], intruder_traj_coordinates[:,1], alpha=0.4, color='g')
          return own_lines, intruder_lines
       
       anim = FuncAnimation(fig, animate, frames=frames, interval=500, repeat=False, blit=True)
    
    if show:
        plt.show()
    
    if animate:
        return anim
    else:
        return None



###############################################################################   


    
def ft_to_m(d):
    return d / 3.28084

def ft_to_km(d):
    return d / 3280.84



###############################################################################   

if __name__ == '__main__' :

    rho=5000
    theta=0.5
    psi=-2
    v_int=500
    v_own=700
 
    print(plt.get_backend())
   
    plot_best_actions(psi=psi, v_own=v_own, v_int=v_int)
    
    plot_best_actions_cartesian(psi=psi, v_own=v_own, v_int=v_int)

    obs = HorizontalObservation(last_a=0, rho=rho, theta=theta, psi=psi, v_int=v_int, v_own=v_own)
    
    plot_state(obs)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8))
    anim = plot_state(obs, animate=True, fig=fig, show=False)
    plt.show()
    