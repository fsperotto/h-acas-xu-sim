# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

import os

import numpy as np

import pygame as pg
#from pygame import gfxdraw


################################################

class AcasPyGameRender():

    def __init__(self, env, 
                 height:int=800, width:int=1200, 
                 margin_factor=0.2,
                 fps=20,
                 idx_own=0, idx_intruder=1):

       self.env = env
           
       self.window_height, self.window_width = height, width
       self.window_size = (width, height)  # Size of the Pygame window
       self.fps = fps
       
       self.idx_own = idx_own
       self.idx_intruder = idx_intruder

       self.own = self.env.airplanes[self.idx_own]
       self.intruder = self.env.airplanes[self.idx_intruder]
       
       
       #own displacement in ft in given max_steps (supposing 1 step = 1 second)
       d_own = self.own.speed * self.env.default_max_steps 
       self.own_dest_x = self.own.x + d_own * np.cos(self.own.head)
       self.own_dest_y = self.own.y + d_own * np.sin(self.own.head)
       self.own_orig_x = self.own.x
       self.own_orig_y = self.own.y

       #int displacement in ft
       d_int = self.intruder.speed * self.env.default_max_steps 
       self.int_dest_x = self.intruder.x + d_int * np.cos(self.intruder.head)
       self.int_dest_y = self.intruder.y + d_int * np.sin(self.intruder.head)
       self.int_orig_x = self.intruder.x
       self.int_orig_y = self.intruder.y
       
       min_x = min(self.own_dest_x, self.own_orig_x, self.int_dest_x, self.int_orig_x)
       max_x = max(self.own_dest_x, self.own_orig_x, self.int_dest_x, self.int_orig_x)
       min_y = min(self.own_dest_y, self.own_orig_y, self.int_dest_y, self.int_orig_y)
       max_y = max(self.own_dest_y, self.own_orig_y, self.int_dest_y, self.int_orig_y)

       ampli_x = max_x - min_x
       ampli_y = max_y - min_y

       #center_x = ampli_x + min_x
       #center_y = ampli_y + min_y
       
       #self.scale_factor = min( self.window_width/ampli_x, self.window_height/ampli_y ) / 1.0
       scale_x =  self.window_width / ampli_x / (1.0 + margin_factor)
       scale_y =  self.window_height / ampli_y / (1.0 + margin_factor)
       self.scale_factor = min( scale_x, scale_y)
           
       #self.offset_height = center_y * self.scale_y  # self.scale_factor 
       #self.offset_width = center_x * self.scale_x  # self.scale_factor 
       self.offset_height = self.window_height * (margin_factor / 2) - (min_y * self.scale_factor)
       self.offset_width = self.window_height * (margin_factor / 2) - (min_x * self.scale_factor)

       pg.init()
       
       pg.event.set_allowed([pg.QUIT])
                             #pg.ACTIVEEVENT, 
                             #pg.KEYDOWN, 
                             #pg.KEYUP,
                             #MOUSEMOTION,
                             #MOUSEBUTTONUP,
                             #MOUSEBUTTONDOWN,
                             #JOYAXISMOTION,
                             #JOYBALLMOTION,
                             #JOYHATMOTION,
                             #JOYBUTTONUP,
                             #JOYBUTTONDOWN,
                             #pg.VIDEORESIZE,
                             #pg.VIDEOEXPOSE,
                             #USEREVENT
                             

       self.font = pg.font.Font(None, 30)
       self.screen = pg.display.set_mode(self.window_size)
       pg.display.set_caption("ACAS-Xu Simulation")

       #self.clock = pg.time.Clock()

       airplane_image_path = os.path.dirname(os.path.realpath(__file__)) + '/img/airplane.png'
       self.airplane_image = pg.image.load(airplane_image_path).convert_alpha()
       self.airplane_image = pg.transform.scale(self.airplane_image, (1800*self.scale_factor, 1800*self.scale_factor))


    def scale(self, coords):
        if isinstance(coords, tuple):
            #return int(coords[0] * self.scale_factor + self.offset_width), self.window_height - int(coords[1] * self.scale_factor + self.offset_height)
            #return int(coords[0] * self.scale_factor + self.offset_width), int(coords[1] * self.scale_factor + self.offset_height)
            return int(coords[0] * self.scale_factor + self.offset_width), self.window_height - int(coords[1] * self.scale_factor + self.offset_height)
        else:
            return [self.scale(coord) for coord in coords]


    def refresh(self):
        
        #if self.first_step:  # Cleaning the screen after each episode
         #   self.screen.fill((255, 255, 255))  # Put a white screen
         #   self.first_step = False

        if pg.event.peek(pg.QUIT):
            
            self.env.renderer = None
            self.close()
            #pg.quit()
            #self.env.close()
            
        else:
            
            pg.draw.lines(self.screen, "gray", False, [self.scale((self.own_orig_x, self.own_orig_y)), self.scale((self.own_dest_x,self.own_dest_y))])
            pg.draw.lines(self.screen, "gray", False, [self.scale((self.int_orig_x, self.int_orig_y)), self.scale((self.int_dest_x,self.int_dest_y))])
    
            if len(self.env.states_history) >= 2:
                palette = ['gray', 'green', 'blue', 'magenta', 'red']
                for i in range(1, len(self.env.states_history)):
                    H1 = self.env.states_history[i-1]
                    H2 = self.env.states_history[i]
                    A = self.env.commands_history[i-1]
                    for idx in [self.idx_own, self.idx_intruder]:
                        coords_pre = self.scale((H1[idx][0], H1[idx][1]))
                        coords_pos = self.scale((H2[idx][0], H2[idx][1]))
                        color = palette[A[idx]]
                        pg.draw.lines(self.screen, color, False, [coords_pre, coords_pos], width=3)   #aalines
                #for idx in [self.idx_own, self.idx_intruder]:
                #    traj = [self.scale((H[idx][0], H[idx][1])) for H in self.env.states_history]   #[(x,y), ...]
                #    pg.draw.lines(self.screen, "black", False, traj)   #aalines
            
            #initial_own = (own.x , own.y)
            #initial_intruder = (intruder.x, intruder.y)
    
            #scale = 0.005  # Adjust the scale compared to the window size
            own_pos = self.scale((self.own.x , self.own.y))
            intruder_pos = self.scale((self.intruder.x, self.intruder.y))
    
            rotated_img_own = pg.transform.flip( pg.transform.rotate(self.airplane_image, -np.degrees(self.own.head)-90), False, True)
            rotated_img_int = pg.transform.flip( pg.transform.rotate(self.airplane_image, -np.degrees(self.intruder.head)-90), False, True)
    
            rect_own = rotated_img_own.get_rect(center=own_pos)
            rect_int = rotated_img_int.get_rect(center=intruder_pos)
    
            self.screen.blit(rotated_img_own, rect_own.topleft)
            self.screen.blit(rotated_img_int, rect_int.topleft)
    
            #Display elapsed time at top left of screen
            elapsed_time = self.env.current_step
            time_text = f'Time: {elapsed_time:.2f} s'
            text_surface = self.font.render(time_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10))
    
            #self.draw_dashed_line(self.screen, (0, 0, 0), intruder_pos, (LENGTH/2,WIDTH/2), dash_length=15, width=2)
    
            #pg.draw.circle(self.screen, (0, 0, 255), own_pos, 10)
            #pg.draw.circle(self.screen, (255, 0, 0), intruder_pos, 10)
    
            pg.display.flip()
            #pg.display.update()
    
            #image = pg.surfarray.array3d(self.screen)
            
            self.screen.fill((255, 255, 255))
            #self.clock.tick(self.metadata["render_fps"])
            #self.clock.tick(self.fps)
    
            #return image


    def close(self):
        if self.screen is not None:
            pg.quit()
            self.screen = None

    

   
###############################################################################   

if __name__ == '__main__' :
   
   #import pynput
   #import pyrl
   
   #print(mpl.get_backend())
   
   #run_single_sim()
   #plt.show(block=True)
   
   #random_run()
   #plt.show(block=True)
   
   #smart_random_run()
   #plt.show(block=True)

   
   #num_sims=30
   #seed=3
   #min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['dubins', 'coc'], plot=False)
   #fig, ax = plt.subplots(figsize=(8,8))
   #for d_evo in min_d_evolution:
   #    plt.plot(d_evo)
   #plt.grid()
   #plt.show(block=True)

   #min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['lut', 'coc'], plot=False)
   #fig, ax = plt.subplots(figsize=(8,8))
   #for d_evo in min_d_evolution:
   #    plt.plot(d_evo)
   #plt.grid()
   #plt.show()

   ###############################   
   
   from acasxu_basics import Airplane
   from acasxu_env import HorizontalAcasXuEnv
   from acasxu_agents import RandomAgent, ConstantAgent, DubinsAgent, LutAgent
   from acasxu_episode_simulator import AcasSim
   
   max_steps = 100
   
   airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0), 
                Airplane(x=50000.0, y=50000.0, head=-np.pi/3, speed=780.0)]
   
   env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

   agents = [DubinsAgent(), DubinsAgent()]

   sim = AcasSim(env, agents)

   #online rendering using pygame
   env.renderer = AcasPyGameRender(env)

   sim.reset()
   sim.run(time_delay=0.05)

   env.close()

   ##offline rendering using matplotlib animation
   #fig, ax = plt.subplots(figsize=(8,8))
   #renderer = AcasRender(env)
   #renderer.plot(fig=fig)
   