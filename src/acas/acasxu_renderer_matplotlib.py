# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

###############################################################################

import os

import numpy as np

import importlib.resources

from matplotlib import pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from scipy import ndimage

###############################################################################

from acas.acasxu_basics import ACT_NAMES

###############################################################################

# 0: rho, distance (ft) [0, 60760]
# 1: theta, angle to intruder relative to ownship heading (rad) [-pi,+pi]
# 2: psi, heading of intruder relative to ownship heading (rad) [-pi,+pi]
# 3: v_own, speed of ownship (ft/sec) [100, 1145? 1200?] 
# 4: v_int, speed in intruder (ft/sec) [0? 60?, 1145? 1200?] 


###############################################################################


class AcasRender():

    def __init__(self, traces,
                 fig=None, ax=None,
                 title="ACAS-Xu Simulation",
                 airplane_size=200,  #ft      # ~ 61m   
                 nmac_radius=500,    #ft      # ~ 152m
                 traj_linewidth = 1,
                 show_nmac_circle=True):

       #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
       self.fig = fig
       self.ax = ax
       if ax is None:
           if fig is None:
               self.fig, self.ax = plt.subplots()
               #if len(self.envs) == 1:
               #    axes = [axes]
               self.fig.tight_layout()
           else:
               self.ax = self.fig.gca()

       self.fig.subplots_adjust(bottom=0.2)  #place for buttons

       #verify that traces is a list
       self.traces = traces  if isinstance(traces, list)  else  [traces]
       
       #current shown trace
       self.current_idx = 0
       self.current_trace = None
       
       self.traj_linewidth = traj_linewidth

       self.airplane_size = airplane_size
       
       self.show_nmac_circle = show_nmac_circle
       self.nmac_radius = nmac_radius
       
       self.airplane_img = None
       try:       
           airplane_img_filepath = os.path.dirname(os.path.realpath(__file__)) + '/img/airplane.png'
       except:
           airplane_img_filepath = './img/airplane.png'
           try:       
               self.airplane_img = plt.imread(airplane_img_filepath)
           except:
               try:       
                   img_dir = importlib.resources('acas.img')
                   airplane_img_filepath = Path(img_dir , 'airplane.png')       
                   self.airplane_img = plt.imread(airplane_img_filepath)
               except:
                   self.airplane_img = None

       #for env, ax in zip(self.envs, axes):
            
       self.ax.set_aspect('equal')
       #ax.axis('equal')
    
       self.ax.set_title(title)
       self.ax.set_xlabel('X Position (ft)')
       self.ax.set_ylabel('Y Position (ft)')

       self.palette = np.array(['gray', 'green', 'blue', 'magenta', 'red'])
       self.custom_lines = [Line2D([0], [0], color=c, lw=1) for c in self.palette]

       #text boxes        
       self.time_text = self.ax.text(0.02, 0.98, 'Time: 0 s', horizontalalignment='left', fontsize=12, verticalalignment='top', transform=self.ax.transAxes)
       self.time_text.set_visible(True)

       #lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
       #self.ax.add_collection(lc)

       self.artists = []

       self.anim = None
       
       self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
       
       # buttons
       ax_prev = self.fig.add_axes([0.25, 0.05, 0.2, 0.075])
       ax_next = self.fig.add_axes([0.55, 0.05, 0.2, 0.075])
       self.btn_prev = Button(ax_prev, '< Prev', )
       self.btn_prev.on_clicked(self.prev_anim_btn_click)
       self.btn_next = Button(ax_next, 'Next >')
       self.btn_next.on_clicked(self.next_anim_btn_click)       

    
    def _prepare_anim(self):

        #get traces of current environment to animate
        self.current_trace = self.traces[self.current_idx]

        #prepare trace data
        self.total_time = self.current_trace.num_steps
        self.num_airplanes = self.current_trace.num_airplanes
        self.H = np.array(self.current_trace.states_history)
        self.A = np.array(self.current_trace.commands_history)
        self.C = self.palette[self.A]

        self.ax.clear()
        
        while len(self.ax.artists) > 0:
            self.ax.artists.remove()

        while len(self.ax.collections) > 0:
            self.ax.collections.remove()
        
        self.ax.legend(self.custom_lines, ACT_NAMES, fontsize=12)#, loc='lower left')

        #trajectories
        self.lines = []
        self.lcs = []
        self.trajs = []
        
        self.nmac_circles = []

        #airplanes
        self.airplane_img_boxes = []
        
        
        for i in range(self.num_airplanes):
            
            #initial state
            x, y, theta, speed = self.H[0,i]
            x_f, y_f, theta_f, speed_f = self.H[-1,i]
            
            #total displacement
            d = speed * self.total_time
            dx = d * np.cos(theta)
            dy = d * np.sin(theta)
            
            # draw linear initial trajectory
            self.ax.scatter([x, x_f], [y, y_f], zorder=3, alpha=0.3)    #final position
            self.ax.plot([x, x+dx], [y, y+dy], ls=':', lw=1, color='lightgray', zorder=2)   #initial linear trajectory
            
            #trajectory using one line per airplane
            #l = ax.plot(x, y, ls='-', lw=1, zorder=1)[0]
            ##l = Line2D(x, y, lc='c', lw=1, ls='-', zorder=1)
            #lines.append(l)
            
            #trajectories using line collections (allow colors)
            #lc = LineCollection([], linestyle='solid', lw=self.traj_linewidth, animated=True, color='k', zorder=1)
            lc = LineCollection([], linestyle='solid', lw=self.traj_linewidth, color='k', zorder=1)
            self.ax.add_collection(lc)
            self.lcs.append(lc)

            #trajectories using scatter
            #traj_points = self.ax.scatter(H[:,i,0], H[:,i,1], color='k', zorder=1)
            #traj_points = self.ax.scatter([], [], color='k', zorder=1)
            #self.trajs.append(traj_points)
            #self.ax.add_collection(lc)
            #lcs.append(lc)
            
            #draw airplanes
            box = Bbox.from_bounds(x - self.airplane_size/2, y - self.airplane_size/2, self.airplane_size, self.airplane_size)
            tbox = TransformedBbox(box, self.ax.transData)
            box_image = BboxImage(tbox, zorder=2)
            self.airplane_img_boxes.append(box_image)
            theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
            
            if self.airplane_img is not None:
                img_rotated = ndimage.rotate(self.airplane_img, theta_deg, order=1)
                box_image.set_data(img_rotated)
                self.ax.add_artist(box_image)
            
            if self.show_nmac_circle:
                nmac_circle = plt.Circle((x, y), self.nmac_radius, color='r', fill=False)
                self.nmac_circles.append(nmac_circle)
                self.ax.add_artist(nmac_circle)
                self.ax.add_patch(nmac_circle)
            
        self.artists = self.airplane_img_boxes + self.nmac_circles + self.lines + self.lcs + self.trajs + [self.time_text]



    def _anim_reset(self):

        #reset graphics
        self.time_text.set_text('Time: 0 s')
        for i, lc in enumerate(self.lcs):
            lc.get_paths().clear()

        return self.artists

    

    def _anim_refresh(self, n):

        self.time_text.set_text(f'Time: {n} s')
        
        for i in range(self.num_airplanes):
            
            #trajectory using lines
            #l = lines[i]
            #l.set_xdata(H[:n+1,i,0])
            #l.set_ydata(H[:n+1,i,1])
            
            #trajectory using paths
            lc = self.lcs[i]
            paths = lc.get_paths()
            paths.append(Path([self.H[n,i,:2], self.H[n+1,i,:2]]))
            ##lc.set_paths(H[:n+1,i,:2])
            lc.set_colors(self.C[:n+1,i])
            
            #trajectory using scatter
            #traj_points = trajs[i]
            #traj_points.set_offsets(H[:n+1,i,:2])  #positions
            #traj_points.set_colors(C[:n+1,i])       #colors
            
            x = self.H[n+1,i,0]
            y = self.H[n+1,i,1]
            theta = self.H[n+1,i,2]
            theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
            
            #airplane image
            self.box_image=self.airplane_img_boxes[i]
            
            if self.airplane_img is not None:
                original_size = list(self.airplane_img.shape)
                img_rotated = ndimage.rotate(self.airplane_img, theta_deg, order=1)
                rotated_size = list(img_rotated.shape)
                ratios = [r / o for r, o in zip(rotated_size, original_size)]
                self.box_image.set_data(img_rotated)
                width = self.airplane_size * ratios[0]
                height = self.airplane_size * ratios[1]
                box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
                tbox = TransformedBbox(box, self.ax.transData)
                self.box_image.bbox = tbox
            
            if self.show_nmac_circle:
                nmac_circle = self.nmac_circles[i]
                nmac_circle.center = (x, y)
                
        return self.artists


    def _next_anim(self):
        if self.current_idx < len(self.traces)-1:
            self.anim.event_source.stop()
            self.current_idx += 1
            self.plot()
        
    def _prev_anim(self):
        if self.current_idx > 0:
            self.anim.event_source.stop()
            self.current_idx -= 1
            self.plot()

    # Buttons
    def next_anim_btn_click(self, event):
        self._next_anim()

    def prev_anim_btn_click(self, event):
        self._prev_anim()

    # Keyboard
    def on_keypress(self, event):
        #DEFAULTS:
        #Home/Reset 	h or r or home
        #Back 	        c or left arrow or backspace
        #Forward 	    v or right arrow
        #Pan/Zoom 	    p
        #Zoom-to-rect 	o
        #Save 	        s or ctrl + s
        #Toggle fullscreen 	f or ctrl + f
        #Close plot 	ctrl + w
        #Close all plots 	shift + w
        #Constrain pan/zoom to x axis 	hold x when panning/zooming with mouse
        #Constrain pan/zoom to y axis 	hold y when panning/zooming with mouse
        #Preserve aspect ratio 	hold CONTROL when panning/zooming with mouse
        #Toggle major grids 	g when mouse is over an axes
        #Toggle minor grids 	G when mouse is over an axes
        #Toggle x axis scale (log/linear) 	L or k when mouse is over an axes
        #Toggle y axis scale (log/linear) 	l when mouse is over an axes            
        
        if event.key == 'z':
            if self.anim._is_paused:
                self.anim.resume()
            else:
                self.anim.pause()
            self.anim._is_paused = not self.anim._is_paused
        elif event.key == ' ':
            if self.anim._is_paused:
                self.anim._step()
        elif event.key == 'n':   #next env, if many
            self._next_anim()
        elif event.key == 'b':   #prev env, if many
            self._prev_anim()
        elif event.key == 'q':
            self.anim.event_source.stop()
            #self.anim.pause()
            #ax.remove()



    def plot(self, interval=20, show=True, block=True, save_as=None, blit=False, repeat=True):
 
        self._prepare_anim()
        
        self.anim = FuncAnimation(self.fig, self._anim_refresh, init_func=self._anim_reset, frames=self.total_time-1, interval=interval, blit=blit, repeat=repeat, repeat_delay=100)
        self.anim._is_paused = False
        
        self.fig.canvas.draw_idle()
        
        #def on_click(event):
        #   if self.anim is not None:
        #      if self.anim._is_paused:
        #          self.anim.resume()
        #      else:
        #          self.anim.pause()
        #      self.anim._is_paused = not self.anim._is_paused
        #   #plt.close(fig)
        #fig.canvas.mpl_connect('button_press_event', on_click)
        
#        def on_close(event):
#            plt.close()
#            if anim is not None:
#                #self.my_anim.pause()
#                anim = None    #setting to None allows garbage collection and closes loop
#                del(anim)
#        fig.canvas.mpl_connect('close_event', on_close)

                        
        if isinstance(save_as, str):
            if save_as[-4:] in [".mp4", ".mpg", ".mpeg"]:
                writer = animation.writers['ffmpeg'](fps=1000//interval, bitrate=1800)
                self.anim.save(save_as, writer=writer)
            elif save_as[-4:] in [".gif"]:
                writer = animation.PillowWriter(fps=1000//interval, bitrate=1800)
                self.anim.save(save_as, writer=writer)
        
        if show:
            #plt.show()
            #plt.draw()
            #plt.draw_idle()
            plt.show(block=block)

        return self.anim
        # self.anim = None


   
###############################################################################   

if __name__ == '__main__' :
   
   from acasxu_basics import Airplane, create_random_incident, HorizontalObservation
   from acasxu_env import HorizontalAcasXuEnv
   from acasxu_episode_simulator import AcasSim
   from acasxu_agents import RandomAgent, ConstantAgent, UtilityModelAgent

   from acasxu_model_lut import NearestLUT
   from acasxu_model_onnx_dubins import DubinsNN

   import matplotlib 
   
   print(matplotlib.rcsetup.interactive_bk)
   matplotlib.use('TkAgg')
   
   print("Executing 1st simulation...")   
   
   max_steps = 100
   
   #own = Airplane(x=13.7, y=54.4, head=2.94, speed=730)
   #intruder = Airplane(x=14.3, y=39.5, head=6.03, speed=638)
   
   own = Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0)
   intruder = Airplane(x=50000.0, y=50000.0, head=-np.pi/3, speed=780.0)
   
   airplanes = [own, intruder]
   
   env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

   agents = [UtilityModelAgent(model=NearestLUT()), UtilityModelAgent(model=NearestLUT())]
   #agents = [UtilityModelAgent(model=DubinsNN()), UtilityModelAgent(model=DubinsNN())]

   sim = AcasSim(env, agents)
   sim.reset()
   sim.run()
   
   trace = env.get_history()

   print("Offline (post-simulation) rendering (using history)...")   

   #offline rendering using matplotlib animation
   fig, ax = plt.subplots(figsize=(8,8))
   renderer = AcasRender(trace, fig=fig, ax=ax)
   renderer.plot()

   print("Executing smart random simulations...")   

   max_steps = 50
   
   traces = []
   
   for i in range(5):

       own, intruder = create_random_incident(incident_time=max_steps//2)
       airplanes = [own, intruder]
       env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)
    
       agents = [ConstantAgent(), UtilityModelAgent(model=NearestLUT())]
    
       sim = AcasSim(env, agents)
       sim.reset()
       sim.run()
       
       traces.append(env.get_history())
    
       agents = [ConstantAgent(), UtilityModelAgent(model=DubinsNN())]

       sim = AcasSim(env, agents)
       sim.reset()
       sim.run()

       traces.append(env.get_history())
    

   print("Offline (post-simulation) rendering (using history)...")   
    
   #offline rendering using matplotlib animation
   fig, ax = plt.subplots(figsize=(8,8))
   renderer = AcasRender(traces, fig=fig, ax=ax)
   renderer.plot(interval=10)
   #renderer.plot(fig=fig, ax=ax, interval=10, title="ONNX")   # save_as="sim.gif"
