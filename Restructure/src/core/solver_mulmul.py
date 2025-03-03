import numpy as np
from src.core.simulate_env import Environment
from typing import Dict

class Solver:
    """Solver for positioning using LEDs and PD signals."""
    
    def __init__(self, environment: Environment, config: Dict):
        self.env = environment
        self.led_system = environment.led_system
        self.pd_system = environment.pd_system
        self.testp = environment.testp

        self.weight_form = config["application"]["weight_form"]
        self.tolerance = config["environment"]["tolerance"]
        self.effective = config["environment"]["effective"]
        self.threshold = config["environment"]["threshold"] 
        
        # Frequent used layered variables
        # self.kpos = self.testp.kpos
        # self.krot = self.testp.krot
        self.led_num = self.led_system.num
        self.pd_num = self.pd_system.num
        
        # Results
        self.ledu = None  # usable led num(int)
        self.pdu = None  # usable pd num(int)
        self.ori_sol_pd_coor = None # kpos, krot, 3
        self.ori_sol_led_coor = None # kpos, krot, 3
        self.sol_dis_av = None # kpos, krot
        self.error = None # kpos, krot
    
    def solve_mulmul(self):
        """
        Solve the positioning problem for given test points.
        """
        
        # [kpos x krot x led_num x pd_num]
        light_f = self.env.light_output_filtered

        # Check which LEDs/PDs have enough valid signals
        # 3d data, not the ledu and pdu for amount
        led_usable = np.sum(~np.isnan(light_f), axis=3) > 2  # [kpos x krot x led_num]
        pd_usable = np.sum(~np.isnan(light_f), axis=2) > 2   # [kpos x krot x pd_num]
        
        # Mask unusable LEDs/PDs
        # [kpos x krot x led_num x pd_num]
        light_led = np.ma.masked_array(
            light_f,
            np.tile(~led_usable, (self.pd_num, 1, 1, 1)).transpose(1, 2, 3, 0)
        )
        light_pd = np.ma.masked_array(
            light_f,
            np.tile(~pd_usable, (self.led_num, 1, 1, 1)).transpose(1, 2, 0, 3)
        )
        
        # Store usable counts
        self.ledu = led_usable.sum(axis=2)  # [kpos x krot]
        self.pdu = pd_usable.sum(axis=2)    # [kpos x krot]
        
        # Get surface normal vectors
        nor_led, nor_pd, conf_led_ref, conf_pd_ref, led_data_other, pd_data_other = self._get_surface(light_led, light_pd)
        
        # Calculate possible position solutions
        cross_led, cross_pd = self._get_cross(
            led_data_other, pd_data_other, light_led, light_pd, 
            nor_led, nor_pd, conf_led_ref, conf_pd_ref
        )
        
        # Calculate final position based on weight method
        if self.weight_form == 'weight':
            # Weight by light intensity
            mask_led = np.isnan((cross_led[:, :, :, :, 0]).filled(fill_value=np.nan))
            weight_led = np.ma.array(light_f, mask=mask_led).filled(fill_value=np.nan)
            total_led = np.nansum(weight_led, axis=(2, 3))
            total_led = np.tile(total_led, (1, 1, 1, 1)).transpose(2, 3, 0, 1)
            weight_led = np.divide(weight_led, total_led)
            weight_led = np.tile(weight_led, (1, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
            sol_led = np.multiply(weight_led, cross_led)
            # kpos, krot, 3
            self.ori_sol_pd_coor = np.nansum(sol_led, axis=(2, 3))
            
            mask_pd = np.isnan((cross_pd[:, :, :, :, 0]).filled(fill_value=np.nan))
            weight_pd = np.ma.array(light_f, mask=mask_pd).filled(fill_value=np.nan)
            total_pd = np.nansum(weight_pd, axis=(2, 3))
            total_pd = np.tile(total_pd, (1, 1, 1, 1)).transpose(2, 3, 0, 1)
            weight_pd = np.divide(weight_pd, total_pd)
            weight_pd = np.tile(weight_pd, (1, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
            sol_pd = np.multiply(weight_pd, cross_pd)
            # kpos, krot, 3
            self.ori_sol_led_coor = np.nansum(sol_pd, axis=(2, 3))
            
        elif self.weight_form == 'mean':
            # Simple average
            self.ori_sol_pd_coor = np.nanmean(cross_led, axis=(2, 3))
            self.ori_sol_led_coor = np.nanmean(cross_pd, axis=(2, 3))
        
        # Calculate incident and output angles
        # [kpos x krot x pd_num]
        sol_in_ang = np.arccos(np.inner(self.ori_sol_pd_coor, self.pd_system.ori_car.T))  
        # [kpos x krot x led_num]
        sol_out_ang = np.arccos(np.inner(self.ori_sol_led_coor, self.led_system.ori_car.T))  
        
        # Calculate distance
        const = self.pd_system.respon * self.pd_system.area * self.led_system.pt * (self.led_num + 1) / (2 * np.pi)
        sol_dis = np.sqrt(const * np.divide(
            np.multiply(
                np.tile(np.power(np.cos(sol_in_ang), self.pd_system.m), (self.led_num, 1, 1, 1)).transpose(1, 2, 0, 3),
                np.tile(np.power(np.cos(sol_out_ang), self.led_system.m), (self.pd_num, 1, 1, 1)).transpose(1, 2, 3, 0)
            ),
            light_f
        )) # kp, kr, l, p
        # print(sol_dis)
        # Calculate average distance and error
        self.sol_dis_av = np.nanmean(sol_dis, axis=(2, 3))  # [kpos x krot]
        
        # Calculate position error
        glob_led_pos = self.env.global_testp_after_rot(self.led_system.pos, self.env.testp.testp_rot)
        glob_led_pos = self.env.global_testp_trans(glob_led_pos, self.env.testp.testp_pos)
        
        self.error = np.sum(np.square(
            np.multiply(self.ori_sol_pd_coor, self.sol_dis_av.reshape(self.env.testp.kpos, -1, 1)) - glob_led_pos[:, :, 0, :]
        ), axis=2)
        
       
        #     'ori_sol_pd_coor': self.ori_sol_pd_coor,
        #     'ori_sol_led_coor': self.ori_sol_led_coor

    


    def _get_surface(self, light_led, light_pd):
        """
        Calculate surface normals for LED and PD systems.
        args:
            light_led, light_pd: light with unusable as nan  # [kpos x krot x led_num x pd_num]
        
        This is an internal helper method for the solve method.
        """
        led_num = self.led_system.num
        pd_num = self.pd_system.num
        kpos = self.testp.kpos
        krot = self.testp.krot
        led_ori_car = self.led_system.ori_car
        pd_ori_car = self.pd_system.ori_car


        # Get reference points (maximum light intensity)
        ref1_led = np.nanargmax(light_led, axis=3)  # [kpos x krot x led_num]
        ref1_pd = np.nanargmax(light_pd, axis=2)    # [kpos x krot x pd_num]
        
        # Create masks for reference points
        maskled = np.full(light_led.shape, False)
        maskled[
            np.repeat(np.arange(kpos), krot * led_num),
            np.tile(np.repeat(np.arange(krot), led_num), kpos),
            np.tile(np.arange(led_num), kpos * krot),
            ref1_led.flatten()
        ] = True  # [kpos x krot x led_num x pd_num]
        
        maskpd = np.full(light_pd.shape, False)
        maskpd[
            np.repeat(np.arange(kpos), krot * pd_num),
            np.tile(np.repeat(np.arange(krot), pd_num), kpos),
            ref1_pd.flatten(),
            np.tile(np.arange(pd_num), kpos * krot)
        ] = True  # [kpos x krot x led_num x pd_num]
        
        # Extract reference and other data
        #  kpos, krot, led_num, 1
        led_data_ref = light_led.copy()
        led_data_ref.mask = (light_led.mask | ~maskled)
        led_data_ref = np.sort(led_data_ref, axis=3)[:, :, :, 0].reshape(kpos, krot, led_num, 1)
        #  kpos, krot, led_num, pd_num
        led_data_other = light_led.copy()
        led_data_other.mask = (light_led.mask | maskled)
        
        #  kpos, krot, 1, pd_num
        pd_data_ref = light_pd.copy()
        pd_data_ref.mask = (light_pd.mask | ~maskpd)
        pd_data_ref = np.sort(pd_data_ref, axis=2)[:, :, 0, :].reshape(kpos, krot, 1, pd_num)
        #  kpos, krot, led_num, pd_num
        pd_data_other = light_pd.copy()
        pd_data_other.mask = (light_pd.mask | maskpd)
        

        # Calculate ratios
        # kpos, krot, led_num, pd_num
        ratio_led = np.power(np.divide(led_data_ref, led_data_other), 1/self.pd_system.m)
        ratio_pd = np.power(np.divide(pd_data_ref, pd_data_other), 1/self.led_system.m)
        
        # Extract hardware orientations
        # kpos, krot, led_num, 1, 3
        conf_led = np.tile(pd_ori_car.T, (kpos, krot, led_num, 1, 1))
        conf_led_ref = np.sort(
            (np.ma.masked_array(conf_led, np.tile((light_led.mask | ~maskled), (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0))),
            axis=3
        )[:, :, :, 0, :].reshape(kpos, krot, led_num, 1, 3)   
        
        conf_led_other = np.ma.masked_array(
            conf_led,
            np.tile(led_data_other.mask, (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
        )

        conf_pd = np.tile(led_ori_car, (kpos, krot, pd_num, 1, 1)).transpose(0, 1, 4, 2, 3)
        conf_pd_ref = np.sort(
            (np.ma.masked_array(conf_pd, np.tile((light_pd.mask | ~maskpd), (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0))),
            axis=2
        )[:, :, 0, :, :].reshape(kpos, krot, 1, -1, 3)
        
        conf_pd_other = np.ma.masked_array(
            conf_pd,
            np.tile(pd_data_other.mask, (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
        )

        # Calculate normal vectors
        # kpos, krot, led_num, pd_num, 3
        nor_led = conf_led_ref - np.multiply(ratio_led.reshape(kpos, krot, led_num, -1, 1), conf_led_other)
        nor_pd = conf_pd_ref - np.multiply(ratio_pd.reshape(kpos, krot, led_num, -1, 1), conf_pd_other)
        
        return nor_led, nor_pd, conf_led_ref, conf_pd_ref, led_data_other, pd_data_other
    
    def _get_cross(self, led_data_other, pd_data_other, light_led, light_pd, nor_led, nor_pd, conf_led_ref, conf_pd_ref):
        """
        Calculate cross product vectors for position determination.
        
        This is an internal helper method for the solve method.
        """

        led_num = self.led_system.num
        pd_num = self.pd_system.num
        kpos = self.testp.kpos
        krot = self.testp.krot
 
        # Get second reference points
        ref2_led = np.nanargmax(led_data_other, axis=3)
        ref2_pd = np.nanargmax(pd_data_other, axis=2)
        
        # Create masks for second reference points
        # kpos, krot, led_num, pd_num
        maskled2 = np.full(light_led.shape, False)
        maskled2[
            np.repeat(np.arange(kpos), krot * led_num),
            np.tile(np.repeat(np.arange(krot), led_num), kpos),
            np.tile(np.arange(led_num), kpos * krot),
            ref2_led.flatten()
        ] = True
        
        maskpd2 = np.full(light_pd.shape, False)
        maskpd2[
            np.repeat(np.arange(kpos), krot * pd_num),
            np.tile(np.repeat(np.arange(krot), pd_num), kpos),
            ref2_pd.flatten(),
            np.tile(np.arange(pd_num), kpos * krot)
        ] = True
        
        # Extract reference and other normal vectors
        # kpos, krot, led_num, 1, 3
        nor_led_ref = nor_led.copy()
        nor_led_ref.mask = np.tile((light_led.mask | ~maskled2), (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
        nor_led_ref = np.sort(nor_led_ref, axis=3)[:, :, :, 0, :].reshape(kpos, krot, led_num, 1, 3)
        # kpos, krot, led_num, pd_num, 3
        nor_led_other = nor_led.copy()
        nor_led_other.mask = (nor_led.mask | np.tile(maskled2, (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0))
        # kpos, krot, 1, pd_num, 3
        nor_pd_ref = nor_pd.copy()
        nor_pd_ref.mask = np.tile((light_pd.mask | ~maskpd2), (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
        nor_pd_ref = np.sort(nor_pd_ref, axis=2)[:, :, 0, :, :].reshape(kpos, krot, 1, -1, 3)
        # kpos, krot, led_num, pd_num, 3
        nor_pd_other = nor_pd.copy()
        nor_pd_other.mask = (nor_pd.mask | np.tile(maskpd2, (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0))
        
        # Calculate cross products [kpos, krot, led_num, pd_num, 3]
        cross_led = np.ma.masked_array(
            np.cross(np.tile(nor_led_ref, (1, 1, 1, pd_num, 1)), nor_led_other),
            nor_led_other.mask
        )
        cross_pd = np.ma.masked_array(
            np.cross(np.tile(nor_pd_ref, (1, 1, led_num, 1, 1)), nor_pd_other),
            nor_pd_other.mask
        )
        
        # Normalize cross products
        cross_led = np.divide(
            cross_led,
            np.tile(np.sqrt(np.sum(np.square(cross_led), axis=4)), (1, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
        )
        cross_pd = np.divide(
            cross_pd,
            np.tile(np.sqrt(np.sum(np.square(cross_pd), axis=4)), (1, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0)
        )
        
        # Ensure proper direction
        cross_led_mask = np.sum(np.multiply(conf_led_ref, cross_led), axis=4) < 0
        cross_led = np.ma.masked_array(
            np.where(np.tile(cross_led_mask, (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0), -cross_led, cross_led),
            nor_led_other.mask
        )
        cross_pd_mask = np.sum(np.multiply(conf_pd_ref, cross_pd), axis=4) < 0
        cross_pd = np.ma.masked_array(
            np.where(np.tile(cross_pd_mask, (3, 1, 1, 1, 1)).transpose(1, 2, 3, 4, 0), -cross_pd, cross_pd),
            nor_pd_other.mask
        )
        
        
        
        
        # [kpos, krot, led_num, pd_num, 3]
        return cross_led, cross_pd