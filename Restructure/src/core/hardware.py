# -*- coding: utf-8 -*-
"""
Harware classes(LED & PD)
Hardware has attribute and method, which is suitable to use object
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict

class Hardware:
    # Base class for LED and PD
    
    def __init__(self, num):
        """
        Harware initialization, w.r.t. number of hardware element.
        
        Args:
            num: Number of elements in this hardware component
        """
        self.num = num
        self.ori_ang = np.tile(np.array([[0, 0, 1]]).T, (1, num))  # angular representation: 3xnum
        self.ori_car = np.tile(np.array([[0, 0, 0]]).T, (1, num)) # Cartesian representation
        self.pos = np.tile(np.array([[0, 0, 0]]).T, (1, num))  # Position [3xnum]
        self.config_num = 1
    def ori_ang2cart(self, ori_ang) ->np.ndarray:
        """
        Set Cartesian orientation: angular to Cartesian coordinates.
        
        Arg:
            ori_ang: Orientation angles [2xnum] (alpha傾角, beta方位角)
        
        Return:
            Cartesian orientation [3xnum]
        """
        return np.stack((
            np.multiply(np.sin(ori_ang[0, :]), np.cos(ori_ang[1, :])),
            np.multiply(np.sin(ori_ang[0, :]), np.sin(ori_ang[1, :])),
            np.cos(ori_ang[0, :])
        ), 0)
    
    def set_config(self, config_num: int, alpha: float) -> None:
        """
        Set hardware config based on configuration number and alpha parameter.
        
        Arg
            config_num: Configuration number 
                0: radial 放射狀
                1: radial with center 放射狀，一個放中間
            alpha: Alpha(rad) 方位角
        """
        self.config_num = config_num
        if config_num == 0:  # Radial pattern
            beta = np.deg2rad(360/self.num)  # Azimuth angle
            self.ori_ang = np.stack((
                alpha * np.ones(self.num),
                (beta * np.arange(1, self.num + 1))
            ), 0)  # 2xnum
            self.ori_car = self.ori_ang2cart(self.ori_ang)  # 3xnum
        
        elif config_num == 1:  # Radial pattern with one in center
            beta = np.deg2rad(360 / (self.num - 1))  # 方位角
            self.ori_ang = np.stack((
                alpha * np.ones(self.num - 1),
                (beta * np.arange(1, self.num))
            ), 0)  # 2x(num-1)
            self.ori_ang = np.concatenate((self.ori_ang, np.array([[0, 0]]).T), axis=1)
            self.ori_car = self.ori_ang2cart(self.ori_ang)  # 3xnum
        self.pos = np.tile(np.array([[0, 0, 0]]).T, (1, self.num))
        # self.config_num = config_num
        ''' Dual ring with more arguments set
        elif config_num == 2:  # Dual ring 
            a = int(self.num * 0.4) # 內圈數量
            beta = np.deg2rad(360 / a)
            ori_anga = np.stack((
                alpha * np.ones(a),
                (beta * np.arange(1, a + 1))
            ), 0)  # 2xa
            
            remaining = self.num - a # 外圈數量
            beta = np.deg2rad(360 / remaining) 
            ori_angb = np.stack((
                3 * alpha * np.ones(remaining),
                (beta * np.arange(1, remaining + 1))
            ), 0)  # 2x(remaining)
            
            self.ori_ang = np.concatenate((ori_anga, ori_angb), axis=1)
            self.ori_car = self.ori_ang2cart(self.ori_ang)  # 3xnum
        '''

class LEDSystem(Hardware):
    # LED system
    
    def __init__(self, num: int, hard_num: int, config_num: int, alpha: float):
        """
        Initialize LED system.
        
        Args:
            num: Number of LEDs
            hard_num: Hardware specification index
            config_num: Configuration number
            alpha: Alpha angle parameter in radians
        """
        super().__init__(num)
        self.set_hardware(hard_num)
        self.set_config(config_num, alpha)
    
    def set_hardware(self, hard_num: int) -> None:
        """
        Set hardware specifications based on index.
        
        Args:
            hard_num: Hardware specification index
        """

        led_list = [ # pt: LED總輻射通量
            1.7*np.pi, 
            80*10**(-3), 
            1.35,
            1.15
        ]
        self.pt = led_list[hard_num] #LED總輻射通量
        self.m = 1  # Default Lambertian order



class PDSystem(Hardware):
    """Class representing the Photodiode hardware system."""
    
    def __init__(self, num: int, hard_num: int, config_num: int, alpha: float):
        """
        Initialize Photodiode system.
        
        Args:
            num: Number of photodiodes
            hard_num: Hardware specification index
            config_num: Configuration number
            alpha: Alpha angle parameter in radians
        """
        super().__init__(num)
        self.set_hardware(hard_num)
        self.set_config(config_num, alpha)
        self.threshold = 10**(-9)  # Default threshold
        self.m = 1  # Default Lambertian order
    
    def set_hardware(self, hard_num: int) -> None:
        """
        Set hardware specifications based on index.
        
        Args:
            hard_num: Hardware specification index
        """
        pd_list = [
            # [pd_respon, pd_area, NEP, dark_current, shunt, capacitance]
            [0.64, 6*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9, 740*10**(-12)],
            [0.64, 5.7*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9, 680*10**(-12)],
            [0.64, 33*10**(-6), 2*10**(-15), 50*10**(-12), 10*10**9, 4000*10**(-12)],
            [0.64, 100*10**(-6), 2.8*10**(-15), 200*10**(-12), 5*10**9, 13000*10**(-12)],
            [0.38, 36*10**(-6), 3.5*10**(-14), 100*10**(-12), 0.1*10**9, 700*10**(-12)]
        ]
        specs = pd_list[hard_num]
        self.respon = specs[0]       # Responsivity
        self.area = specs[1]         # Effective area
        self.NEP = specs[2]          # Noise equivalent power
        self.dark_current = specs[3] # Dark current
        self.shunt = specs[4]        # Shunt resistance
        self.capacitance = specs[5]  # Capacitance

        self.saturate = 10*10**(-3)  # Default saturation level
    

