import matplotlib.pyplot as plt
import gym
import numpy as np
from numba import njit
from Worm_Env.c_worm import Worm
from typing import List
import numpy.typing as npt

class WormSimulationEnv(gym.Env):
    def __init__(self, pulse_timesteps:List[int] = [0,1,2,3,4,5] ,graphing:bool = False):
        super().__init__()
        self.dimx, self.dimy = 1600.0, 1200.0
        self.episode_len = 100         # ticks 0 … 100
        # ------------------------------------------------
        if graphing:
            self.fig, self.ax = plt.subplots()
        self.pulse = pulse_timesteps
        self.prob = 0
        #self.reward = (self.episode_len)/(self.episode_len-pulse_timesteps[-1])
        self.worm:Worm = Worm(position=[self.dimx / 2.0, self.dimy / 2.0])
        self.reward = 0
        self.reset(self.prob)
        

    @staticmethod
    @njit
    def calculate_rewards(left_speed: float, right_speed: float, prob: int,reward: float,mem_reward:bool) -> float:
        """
        Reward rules
        ------------
        • prob == 1  ➜ reward when the worm turns **left**  
        (right wheel faster → right_speed > left_speed)

        • prob == 0  ➜ reward when the worm turns **right**  
        (left wheel faster → left_speed > right_speed)

        Returns +1.0 on a correct turn, −0.1 otherwise.
        """
        if prob == 1 and right_speed > left_speed and mem_reward:      # turning left
            return 1
        if prob == 0 and left_speed > right_speed:      # turning right
            return 1
        return 0


    def reset(self,prob:int):
        if self.worm:
            del self.worm
        self.worm = Worm(position=[self.dimx / 2.0, self.dimy / 2.0])
        self.step_count = 0
        self.reward=0
        self.prob = prob
        return self._get_observations()

    def step(self, actions):
        self.step_count+=1

        left_speed, right_speed = actions
        self.worm.kinematic_step(left_speed=left_speed, right_speed=right_speed)
        observations = self._get_observations()
        #if self.prob==1:
        #    self.worm.sees_food= True if self.step_count in self.pulse else False
        reward = self.calculate_rewards(left_speed,right_speed,self.prob,self.reward,self.step_count>=self.pulse[-1])
        self.reward += reward
        if reward > 0 and self.step_count < 10:
            print("visca als nubies")
        del left_speed,right_speed
        return observations
    
    def get_pos(self):
        return self.worm.position

    
    def _get_observations(self) ->npt.NDArray:  
        min_distance_to_wall = min(self.worm.position[0], self.dimx - self.worm.position[0],\
                                    self.worm.position[1], self.dimy - self.worm.position[1])
        observation = np.array([
                min_distance_to_wall,
                self.worm.posX,
                self.worm.posY,
                self.worm.facing_dir,
                True if self.step_count in self.pulse else False
                                ])
        return np.array(observation)


    def render(self, mode='human'):
        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        w = self.worm                         # ← just one worm
        self.ax.plot(w.position[0], w.position[1], 'ro')
        self.ax.plot([w.position[0], w.position[0] + 100*np.cos(w.facing_dir)],
                    [w.position[1], w.position[1] + 100*np.sin(w.facing_dir)],
                    'b-')
        self.ax.set_xlim(0, self.dimx)
        self.ax.set_ylim(0, self.dimy)
        self.fig.canvas.draw()               # draw once per call


    def close(self):
        plt.close()