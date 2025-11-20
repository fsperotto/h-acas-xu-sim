# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

import os

import numpy as np

from typing import Iterable, Callable
import pygame as pg
#from pygame import gfxdraw

from acas.acasxu_basics import COC_IDX, WR_IDX, WL_IDX, SR_IDX, SL_IDX

###############################################################################

X = 0
Y = 1

###############################################################################

class ACASPyGameGUI():

    #--------------------------------------------------------------    
    def __init__(self, sim,
                 height=800, width=1000, 
                 fps=80,
                 margin_factor=0.2,         #ACAS
                 idx_own=0, idx_intruder=1, #ACAS
                 batch_run=10000,
                 on_close_listeners:Iterable[Callable]=[],
                 close_on_finish=True, finish_on_close=False):

        ##### PYGAME INIT
        
        self.height = height
        self.width = width

        self.window_height, self.window_width = height, width
        self.window_size = (width, height)  # Size of the Pygame window
        
        self.fps = fps
        self.refresh_interval_ms = max(10, 1000 // self.fps)
        
        self.batch_run = batch_run
        
        self.close_on_finish = close_on_finish
        self.finish_on_close = finish_on_close
        
        self.on_close_listeners = on_close_listeners
        
        self._is_closing = False

        self._is_running = False    #clock state
        
        pg.init()

        self.CLOCKEVENT = pg.USEREVENT+1
        #self.clock = pg.time.Clock()
        
        self.screen = None
        
        #self.sim.add_listener('round_finished', self.on_clock)

        self.font = pg.font.Font(None, 30)
        #self.screen = pg.display.set_mode(self.window_size)

        self.draw_original_trajectories = True
        self.draw_tracks = True

        ##### ACAS INIT

        self.sim = sim
        self.env = self.sim.env
        
        self.idx_own = idx_own
        self.idx_intruder = idx_intruder

        self.own = self.env.airplanes[self.idx_own]
        self.intruder = self.env.airplanes[self.idx_intruder]
        
        #positions
        self.origin_coords = [airplane.coords for airplane in self.env.airplanes]
        #displacements in ft in given max_steps (supposing 1 step = 1 second)
        self.destination_coords = [airplane.get_projected_linear_straight_position(self.env.default_max_steps) for airplane in self.env.airplanes]
        positions = np.array(self.origin_coords + self.destination_coords)
        max_x, max_y = np.max(np.array(positions), axis=0)
        min_x, min_y = np.min(np.array(positions), axis=0)
        
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
        
        self.user_action_keys = {
                                pg.K_KP8   : COC_IDX, #FORWARD
                                pg.K_KP5   : COC_IDX, #FORWARD
                                pg.K_KP6   : WR_IDX,  #WEAK RIGHT
                                pg.K_RIGHT : WR_IDX,  #WEAK RIGHT
                                pg.K_KP4   : WL_IDX,  #WEAK LEFT
                                pg.K_LEFT  : WL_IDX,  #WEAK LEFT
                                pg.K_KP9   : SR_IDX,  #STRONG RIGHT
                                pg.K_KP7   : SL_IDX   #STRONG LEFT
                               }
        self.current_user_action = None
        
    #--------------------------------------------------------------    
    def scale(self, coords):
        if isinstance(coords, tuple):
            #return int(coords[0] * self.scale_factor + self.offset_width), self.window_height - int(coords[1] * self.scale_factor + self.offset_height)
            #return int(coords[0] * self.scale_factor + self.offset_width), int(coords[1] * self.scale_factor + self.offset_height)
            return int(coords[X] * self.scale_factor + self.offset_width), self.window_height - int(coords[Y] * self.scale_factor + self.offset_height)
        else:
            return [self.scale(coord) for coord in coords]
        
    #--------------------------------------------------------------    
    def reset(self):
       
        self._is_closing = False
        self._is_running = False    #clock state
        
        self.current_user_action = None
        
        self.sim.reset()

    #--------------------------------------------------------------    
    def set_timer_state(self, state:bool):
       
        self._is_running = state
        
        if state == True:
           pg.time.set_timer(self.CLOCKEVENT, self.refresh_interval_ms)
        else:
           pg.time.set_timer(self.CLOCKEVENT, 0)

    #--------------------------------------------------------------    
    def launch(self, give_first_step=True, start_running=True, window_caption="ACAS-Xu Simulation"):

        pg.display.init()
        #self.window = pg.display.set_mode( (self.width, self.height) )
        self.screen = pg.display.set_mode( self.window_size )
        #pg.display.set_caption('Exp')        
        #self.window.set_caption('Exp')
        
        pg.display.set_caption(window_caption)

        airplane_image_path = os.path.dirname(os.path.realpath(__file__)) + '/img/airplane.png'
        self.airplane_image = pg.image.load(airplane_image_path).convert_alpha()
        self.airplane_image = pg.transform.scale(self.airplane_image, (1800*self.scale_factor, 1800*self.scale_factor))
        
        if give_first_step:
           self.sim.step()
           self.refresh()
        
        if start_running:
           self.set_timer_state(True)
        
        #ACTIVE
        try:

           while not self._is_closing:
              
              event = pg.event.wait()
              
              self.process_event(event)
              
              if self.close_on_finish and self.env.done :
                 self.close()

        except KeyboardInterrupt:
           self.close()
           print("KeyboardInterrupt: simulation interrupted by the user.")

        except:
           self.close()
           raise

     
    #--------------------------------------------------------------    
    def refresh(self):
        
        if self.draw_original_trajectories:
            for origin, dest in zip(self.origin_coords, self.destination_coords):
                pg.draw.lines(self.screen, "gray", False, [self.scale((origin[X], origin[Y])), self.scale((dest[X], dest[Y]))])
        
        if self.draw_tracks:
            if len(self.env.states_history) >= 2:
                palette = ['black', 'green', 'blue', 'magenta', 'red']
                for i in range(1, len(self.env.states_history)):
                    H1 = self.env.states_history[i-1]
                    H2 = self.env.states_history[i]
                    A = self.env.commands_history[i-1]
                    for i in range(len(self.env.airplanes)):
                        coords_pre = self.scale((H1[i][X], H1[i][Y]))
                        coords_pos = self.scale((H2[i][X], H2[i][Y]))
                        color = palette[A[i]]
                        pg.draw.lines(self.screen, color, False, [coords_pre, coords_pos], width=3)   #aalines
                #for idx in [self.idx_own, self.idx_intruder]:
                #    traj = [self.scale((H[idx][0], H[idx][1])) for H in self.env.states_history]   #[(x,y), ...]
                #    pg.draw.lines(self.screen, "black", False, traj)   #aalines
        
        for airplane in self.env.airplanes:

            #Adjust the scale compared to the window size
            pos = self.scale((airplane.x , airplane.y))

            #Rotate airplane image given heading
            rotated_img = pg.transform.flip( pg.transform.rotate(self.airplane_image, -np.degrees(airplane.head)-90), False, True)

            rect = rotated_img.get_rect(center=pos)

            self.screen.blit(rotated_img, rect.topleft)

        #Display elapsed time at top left of screen
        elapsed_time = self.env.current_step
        time_text = f'Time: {elapsed_time:.2f} s'
        text_surface = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        #self.draw_dashed_line(self.screen, (0, 0, 0), intruder_pos, (LENGTH/2,WIDTH/2), dash_length=15, width=2)

        #pg.draw.circle(self.screen, (0, 0, 255), own_pos, 10)
        #pg.draw.circle(self.screen, (255, 0, 0), intruder_pos, 10)

        pg.display.flip()

        #image = pg.surfarray.array3d(self.screen)
        
        self.screen.fill((255, 255, 255))
        #self.clock.tick(self.metadata["render_fps"])
        #self.clock.tick(self.fps)

        #return image
        
        #refresh
        #pg.display.update()
                
    #--------------------------------------------------------------    
    def close(self):
        
        #if self.screen is not None:
        #    pg.quit()
        #    self.screen = None       
        self._is_closing = True
        
        #stop running
        self.set_timer_state(False)
 
        #if self.window is not None:
        if self.screen is not None:

            #CLOSING
            for callback in self.on_close_listeners:
               callback(self)
               
            pg.display.quit()
            pg.quit()
            
            self.screen = None
           
        #if self.finish_on_close:
        #   self.run()

        #if self.env is not None:
        #   self.env.close()

    #--------------------------------------------------------------    

    def step(self):
        #keys = pg.key.get_pressed()
        #if keys[pg.K_LEFT]:
        #    ...
        if self.current_user_action is not None:
            self.sim.joint_action[self.idx_own] = self.current_user_action
        self.sim.step()
        self.refresh()
        
    #--------------------------------------------------------------    
    def process_event(self, event):
       
         if (event.type == pg.QUIT):
             self.close()
   
         elif event.type == pg.KEYDOWN:
             self.on_keydown(event.key)
             
         elif event.type == self.CLOCKEVENT:
             self.step()
      
    #--------------------------------------------------------------    
    def on_keydown(self, key):
       
       #ESC = exit
       if key == pg.K_ESCAPE:
          self.close()
          
       #P = pause/play
       elif key == pg.K_p:
          self.set_timer_state(not self._is_running)
          
       #R = reset
       elif key == pg.K_r:
          self.set_timer_state(False)
          self.reset()
          self.refresh()
          
       #S = step
       elif key == pg.K_s:
          self.step()
            
       #B = batch run
       elif key == pg.K_b:
          self.sim.run(self.batch_run)
          self.refresh()

       #Q = simulation run
       elif key == pg.K_q:
          self.sim.run()
          self.refresh()

       else:
           if key in self.user_action_keys:
               self.current_user_action = self.user_action_keys[key]
               if not self._is_running:
                   self.step()
                   

   
###############################################################################   

if __name__ == '__main__' :

   from acasxu_basics import Airplane, HorizontalEncounter, AirplanesHorizontalEncounter
   from acasxu_env import HorizontalAcasXuEnv
   from acasxu_agents import RandomAgent, ConstantAgent, DubinsAgent, LutAgent  #UtilityModelAgent
   #from acasxu_model_lut import NearestLUT
   #from acasxu_model_onnx_dubins import DubinsNN
   from acasxu_episode_simulator import AcasSim
   import math
    
   print()
   print("Use keyboard to control simulation:")
   print()
   print(" p = play/pause")
   print(" s = step")
   print(" r = reset")
   print(" b = batch")
   print(" q = run until the end in background")
   print()
   print(" NUM_4 (<--) = WEAK LEFT")
   print(" NUM_6 (<--) = WEAK RIGHT")
   print(" NUM_5 ( | ) = COC")
   print(" NUM_7 (<<-) = STRONG LEFT")
   print(" NUM_9 (->>) = STRONG RIGHT")
   print()
   print(" ESC = quit" )
   print()
   
   #---------------------------------------------------------------------------

   max_steps = 120
   fps = 20
   
   #---------------------------------------------------------------------------
   
   agents = [ConstantAgent(), LutAgent()]

   #---------------------------------------------------------------------------
  
   rho, theta, psi, v_own, v_int = 10000, -math.pi/3, math.pi/3, 200, 100
   
   enc = HorizontalEncounter(v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)
   own, intruder = enc.create_airplanes()
   airplanes = [own, intruder]
   
   env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)
   sim = AcasSim(env, agents)

   gui = ACASPyGameGUI(sim, fps=fps, close_on_finish=False)
   gui.launch(start_running=True, window_caption="Reference Situation")

   #---------------------------------------------------------------------------

   enc = HorizontalEncounter(v_own=v_own, v_int=v_int, theta=-theta, psi=-psi, rho=rho)
   own, intruder = enc.create_airplanes()
   airplanes = [own, intruder]
   
   env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)
   sim = AcasSim(env, agents)

   gui = ACASPyGameGUI(sim, fps=fps, close_on_finish=False)
   gui.launch(start_running=True, window_caption="Inverse Symmetrical")
   
   
   #---------------------------------------------------------------------------
   
   own = Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0)
   intruder = Airplane(x=50000.0, y=50000.0, head=-math.pi/3, speed=780.0)
   airplanes = [own, intruder]

   #enc = AirplanesHorizontalEncounter(own, intruder)

   env = HorizontalAcasXuEnv(airplanes=airplanes, save_states=True, default_max_steps=max_steps)

   sim = AcasSim(env, agents)

   gui = ACASPyGameGUI(sim, fps=fps, close_on_finish=False)
   gui.launch(start_running=True, window_caption="User vs LUT")

   #---------------------------------------------------------------------------
   
   sim.agents = [ConstantAgent(), DubinsAgent()]
   sim.reset()

   gui = ACASPyGameGUI(sim, fps=fps, close_on_finish=False)
   gui.launch(start_running=True, window_caption="User vs LUT")