"""
Positioning solver: multiple LED to mul PD

Implement the algorithm for solving the positioning problem
using mul LEDs and PD.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union, List

from src.core.scenario import TestPoint
from src.core.hardware import LEDSystem, PDSystem

# from src.core.geometry import (
#     global_testp_after_rot, global_testp_trans, interactive_btw_pdled, 
#     filter_view_angle, testp_rot_matlist
# )


class Environment:
    """Solver for positioning using LEDs and PD."""
    
    def __init__(self, led_system: LEDSystem, pd_system: PDSystem, testp: TestPoint, config: Dict):
        self.led_system = led_system
        self.pd_system = pd_system
        self.testp = testp
        self.bandwidth = config["environment"]["bandwidth"]
        self.background = float(config["environment"]["background"])
        self.gain = config["environment"]["gain"]
        self.threshold = float(config["environment"]["threshold"])
        # print('threshold:', self.threshold)

        # result  [kpos x krot x led_num x pd_num]
        self.light_output_filtered = None
        self.simulate_pd_sig()

        


        
    
    
    def simulate_pd_sig(self):
        """
        Solve the positioning problem for given test points.
        
        Args:
            testp_pos: Test point positions [3 x kpos]
            testp_rot: Test point rotations [3 x krot]
        
        Returns:
            Dictionary of results
        """
        kpos = self.testp.kpos
        krot = self.testp.krot
        testp_pos = self.testp.testp_pos
        testp_rot = self.testp.testp_rot
        bandwidth = self.bandwidth

        # Get hardware parameters
        led_num = self.led_system.num
        pd_num = self.pd_system.num
        led_m = self.led_system.m
        pd_m = self.pd_system.m
        led_pt = self.led_system.pt
        pd_respon = self.pd_system.respon
        pd_area = self.pd_system.area
        pd_saturate = self.pd_system.saturate
        dark_current = self.pd_system.dark_current
        shunt = self.pd_system.shunt
        # capacitance = self.pd_system.capacitance
        
        pd_pos = self.pd_system.pos  # PD positions [3 x pd_num]
        pd_ori_car = self.pd_system.ori_car  # PD orientation vectors [3 x pd_num]
        
        led_pos = self.led_system.pos  # LED positions [3 x led_num]
        led_ori_car = self.led_system.ori_car  # LED orientation vectors [3 x led_num]

        # Transform LED positions and orientations to global coordinates
        # ^PL{H}
        # [kpos x krot x led_num x 3(xyz)]
        glob_led_pos = self.global_testp_trans(self.global_testp_after_rot(led_pos, testp_rot), testp_pos)
        # [kpos x krot x 3 x led_num]
        glob_led_ori = np.tile(self.global_testp_after_rot(led_ori_car, testp_rot), (kpos, 1, 1, 1)).transpose((0, 1, 3, 2))


        # Transform PD positions to global coordinates
        # ^LP{H}
        # [krot x 3 x 3]
        glob_inv_pd_pos = self.testp_rot_matlist(testp_rot).transpose(0, 2, 1)
        # [kpos x krot x pd_num x 3]
        glob_inv_pd_pos = (np.tile(glob_inv_pd_pos @ pd_pos,
                                  (kpos, 1, 1, 1)) -
                          np.tile(glob_inv_pd_pos @ testp_pos,
                                 (pd_num, 1, 1, 1)).transpose(3, 1, 2, 0)
                          ).transpose(0, 1, 3, 2)
        
        # Calculate interaction parameters
        # [kpos x krot x led_num x pd_num]
        dis, in_ang, out_ang = self.interactive_btw_pdled(glob_led_pos, glob_led_ori, pd_pos, pd_ori_car)

        # Filter by viewing angles
        pd_view = 2 * np.arccos(np.exp(-np.log(2) / pd_m))
        led_view = 2 * np.arccos(np.exp(-np.log(2) / led_m))
        # with lambertian model's view angle
        in_ang_view = self._filter_view_angle(in_ang, pd_view)
        out_ang_view = self._filter_view_angle(out_ang, led_view)
        # with physical constraint
        in_ang_view[in_ang_view >= np.pi/2] = np.nan
        out_ang_view[out_ang_view >= np.pi/2] = np.nan
        
        # Calculate ideal photocurrent intensity
        const = pd_respon * pd_area * led_pt * (led_num + 1) / (2 * np.pi)
        #  light size: [kpos x krot x led_num x pd_num]
        light = const * np.divide(
            np.multiply(
                np.power(np.cos(in_ang_view), pd_m),
                np.power(np.cos(out_ang_view), led_m)
            ),
            np.power(dis, 2)
        )
        
        # Add noise
        boltz = 1.380649 * 10**(-23)
        temp_k = 300  # Absolute temperature (K)
        elec_charge = 1.60217663 * 10**(-19)
        
        
        # Calculate thermal noise
        thermal_noise = 4 * temp_k * boltz * bandwidth / shunt
        
        # Calculate noise variance
        noise_var = np.sqrt(thermal_noise + 
                          2 * elec_charge * bandwidth * (light + self.background + dark_current))
        
        # Generate random noise
        np.random.seed(10)  # Fixed seed for reproducibility
        noise = np.random.standard_normal(size=light.shape) * noise_var
        
        # Apply gain and add noise [kpos x krot x led_num x pd_num]
        light_with_noise = self.gain * light + noise

        # Filter signals if saturate or below threshold
        # (assume the data be filtered here because saturation should be implemented in simulation)
        # [kpos x krot x led_num x pd_num]
        self.light_output_filtered = self._filter_signals(light_with_noise)
        
        


    def _filter_signals(self, light_signals: np.ndarray) -> np.ndarray:
        """
        Filter signals based on threshold and saturation.
        
        Args:
            light_signals: Raw light signals [kpos x krot x led_num x pd_num]
            
        Returns:
            Filtered light signals [kpos x krot x led_num x pd_num]
        """
        # Create a copy to avoid modifying the original data
        light_f = np.copy(light_signals)
        
        # Filter out signals below threshold or above saturation
        light_f[light_f <= self.threshold] = np.nan
        light_f[light_f >= self.pd_system.saturate] = np.nan
        
        return light_f



    # ==================================================
    def testp_rot_matlist(self, testp_rot: np.ndarray) -> np.ndarray:
        """
        Generate a list of rotation matrices from test point rotation parameters.
        
        Args:
            testp_rot: Test point rotation parameters [3 x krot]
        
        Returns:
            List of rotation matrices [krot x 3 x 3]
        """
        krot = testp_rot.shape[1]
        out = np.zeros((krot, 3, 3))
        
        for i in range(krot):
            out[i, :, :] = self.rotate_mat(testp_rot[:, i])
        
        return out
    
    def global_testp_after_rot(self, pos: np.ndarray, testp_rot: np.ndarray) -> np.ndarray:
        """
        Transform local positions to global coordinates after rotation.
        
        Args:
            pos: Local positions [3 x m]
            testp_rot: Test point rotation parameters [3 x krot]
        
        Returns:
            Global positions after rotation [krot x 3 x m]
        """
        rot_list = self.testp_rot_matlist(testp_rot)  # [krot x 3 x 3]
        return rot_list @ pos  # [krot x 3 x m]
    
    def global_testp_trans(self, pos: np.ndarray, testp_pos: np.ndarray) -> np.ndarray:
        """
        Transform positions after rotation to global coordinates with translation.
        
        Args:
            pos: Positions after rotation [krot x 3 x m]
            testp_pos: Test point positions [3 x kpos]
        
        Returns:
            Global positions after rotation and translation [kpos x krot x m x 3]
        """
        
        return np.tile(pos, (self.testp.kpos, 1, 1, 1)).transpose((0, 1, 3, 2)) + \
               np.tile(testp_pos.T, (self.testp.krot, pos.shape[2], 1, 1)).transpose((2, 0, 1, 3))
    
    def interactive_btw_pdled(self, glob_led_pos: np.ndarray, glob_led_ori: np.ndarray,
                             pd_pos: np.ndarray, pd_ori_car: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate interaction parameters between LEDs and PDs.
        
        Args:
            glob_led_pos: Global LED positions [kpos x krot x led_num x 3]
            glob_led_ori: Global LED orientations [kpos x krot x led_num x 3]
            pd_pos: PD positions [3 x pd_num]
            pd_ori_car: PD orientations [3 x pd_num]
        
        Returns:
            Tuple of (distance, incidence angle, emission angle) arrays
        """
        

        # Calculate position differences
        # kpos, krot, led_num, pd_num, 3
        pos_delta = np.tile(glob_led_pos, (self.pd_system.num, 1, 1, 1, 1)).transpose((1, 2, 3, 0, 4)) - \
                    np.tile(pd_pos.T, (self.testp.kpos, self.testp.krot, self.led_system.num, 1, 1))
        
        # Calculate distances
        dis = np.sqrt(np.sum(np.square(pos_delta), axis=4))  # [kpos x krot x led_num x pd_num]
        
        # Calculate incidence angles
        in_ang = np.arccos(np.divide(
            np.sum(np.multiply(
                np.tile(pd_ori_car.T, (self.testp.kpos, self.testp.krot, self.led_system.num, 1, 1)), 
                pos_delta
            ), axis=4),
            dis
        ))
        
        # Calculate emission angles
        out_ang = np.arccos(np.divide(
            np.sum(np.multiply(
                -pos_delta,
                np.tile(glob_led_ori, (self.pd_system.num, 1, 1, 1, 1)).transpose((1, 2, 3, 0, 4))
            ), axis=4),
            dis
        ))
        
        return dis, in_ang, out_ang
    
    def _filter_view_angle(self, mat: np.ndarray, ang: float) -> np.ndarray:
        """
        Filter out angles that are outside the viewing angle.
        
        Args:
            mat: Matrix of angles [kpos x krot x led_num x pd_num]
            ang: Maximum viewing angle
        
        Returns:
            Filtered matrix with NaN values for angles outside viewing angle
                [kpos x krot x led_num x pd_num]
        """
        mat_view = np.empty_like(mat)
        mat_view[:] = mat
        mat_view[mat_view >= ang] = np.nan
        return mat_view
    
    def rotate_mat(self, ang_list):#ang_list[x,y,z]的角度in rad #順序是先轉x->y->z
        return np.dot(self.rotate_z(ang_list[2]),(np.dot(self.rotate_y(ang_list[1]), self.rotate_x(ang_list[0]))) ) 

    def rotate_x(self, ang): #ang[rad](3*3)
        rot = np.array([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])
        # print(rot)
        return rot #是一個matrix
    def rotate_y(self, ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad](3*3)
        rot = np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
        # print(rot)
        return rot #是一個matrix
    def rotate_z(self, ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad](3*3)
        rot = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
        # print(rot)
        return rot #是一個matrix

    
