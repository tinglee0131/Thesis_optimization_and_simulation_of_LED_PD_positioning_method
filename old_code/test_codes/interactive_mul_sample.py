#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 05:27:56 2022

@author: tiffany
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:43:08 2022

@author: tiffany
"""
from funcfile import *
from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# set environment
def solve_mulmul(led_num,pd_num,led_m,pd_m):
# set environment

    threshold = 0.001
    
    pd_num = int(pd_num)
    led_num = int(led_num)
    pd_m = int(pd_m)
    led_m = int(led_m)
    #pd_num = 7
    #pd_m = 3
    pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    pd_alpha = np.deg2rad(35)#傾角
    pd_beta = np.deg2rad(360/pd_num)#方位角
    
    
    #led_num = 5
    #led_m = 10
    led_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    led_alpha = np.deg2rad(45)#傾角
    led_beta = np.deg2rad(360/led_num)#方位角
    
    pd_area = 1
    led_pt = 1
    pd_saturate = np.inf
    pd_respon = 1
    # =============================================================================
    # ori_tar = np.deg2rad(np.array([[30,20]])).T #2x1
    # ori_tar_cart = ori_ang2cart(ori_tar)#3x1
    # tar_car_correct = ori_tar_cart
    # =============================================================================
    
    # config
    pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
    pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
    pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
    pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3
    
    led_pos = np.tile(np.array([[0,0,0]]).T,(1,led_num))
    led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
    led_ori_car = ori_ang2cart(led_ori_ang) #3xled
    led_rot_mat = rotate_z_mul(led_ori_ang[1,:]) @ rotate_y_mul(led_ori_ang[0,:])#ledx3x3
    
    
    # sample point
    testp_pos = (np.mgrid[-1:1:4j, -1:1:4j, 1:3:4j].reshape(-1,4*4*4)) # 3x?
    # kpos = testp_pos.shape[1]
    testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
    # krot = testp_rot.shape[1]
    # testp_pos = np.array([[0,1,1],[0,0,1],[0,-1,2]]).T # 3x?
    kpos = testp_pos.shape[1]
    # testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
    krot = testp_rot.shape[1]
    
    #(kpos,krot,led_num,3)  
    glob_led_pos = global_testp_trans(global_testp_after_rot(led_pos,testp_rot), testp_pos)
    glob_led_ori = np.tile(global_testp_after_rot(led_ori_car,testp_rot), (kpos,1,1,1)).transpose((0,1,3,2))
    #(kpos,krot,pd_num,3)  
    glob_inv_pd_pos = testp_rot_matlist(testp_rot).transpose(0,2,1)
    glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,\
                                (kpos,1,1,1))\
                        -np.tile(glob_inv_pd_pos@testp_pos\
                                 ,(pd_num,1,1,1)).transpose(3,1,2,0)\
                        ).transpose(0,1,3,2)
    # print(glob_inv_pd_pos)
    
    # print(glob_inv_pd_pos)
    
    # krot,kpos,led_num,pd_num
    dis,in_ang,out_ang = interactive_btw_pdled(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car)
    
    
    # 在view angle外的寫nan
    
    in_ang_view = filter_view_angle(in_ang,pd_view)
    out_ang_view = filter_view_angle(out_ang,led_view)
    
    
    const = pd_respon * pd_area * led_pt * (led_num+1)/(2*np.pi)
    light = const * np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
    # light = np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
    mask_light= np.isnan(light)
    light[mask_light] = 0
    
    # =============================================================================
    # 這裡處理加上noise的部分
    # =============================================================================
    
    boltz = 1.380649 * 10**(-23)
    temp_k = 300
    bandwidth = 300
    elec_charge = 1.60217663 * 10**(-19)
    shunt = 50
    
    noise = np.sqrt(4*temp_k*boltz*bandwidth/shunt\
              + 2*elec_charge*bandwidth*light\
                  ) #+ 2*elec_charge*bandwidth*dark
    # print(noise[0,0,0,0])
    light_noise = light + noise
                  
    
    
    
    # -------以下是硬體部分------------------
    
    
    
    
    # filter掉訊號中小於threshold的部分：nan
    # krot,kpos,led_num,pd_num
    light_f = np.copy(light_noise)
    light_f[light_f <= threshold] = np.nan
    light_f[light_f >= pd_saturate] = pd_saturate
    
    
    
    
    
    # =============================================================================
    # 判斷特定LED是否有>=三個PD接收（才能判斷方位）
    # =============================================================================
    
    
    
    led_usable = np.sum(~np.isnan(light_f),axis=3)>2 #kp,kr,led,
    pd_usable = np.sum(~np.isnan(light_f),axis =2 )>2#kp,kr,pd,
    # pd_usable[2]=False
    # 遮掉unusable
    light_led = np.ma.masked_array(light_f,np.tile(~led_usable,(pd_num,1,1,1)).transpose(1,2,3,0)) #kp,kr,ledu, pd
    light_pd = np.ma.masked_array(light_f, np.tile(~pd_usable,(led_num,1,1,1)).transpose(1,2,0,3))#.reshape(kpos,krot,led_num,-1) #kp,kr,led, pdu
    # print(light_f,"---------")
    
    # =============================================================================
    # 取強度最大者作為ref1_led，建立平面的基準
    # 並利用maskled將light_led分成ref和other
    # => 計算ratio_led
    # =============================================================================
    ledu = led_usable.sum(axis=2)#kp,kr
    pdu = pd_usable.sum(axis=2)#kp,kr
    
    # print('Led, Pd usable amount: ',ledu,pdu)
    ref1_led = np.nanargmax(light_led, axis = 3) #kp,kr,ledu,
    ref1_pd = np.nanargmax(light_pd, axis = 2) #kp,kr,pdu,
    # =============================================================================
    # print(light_led)
    # print(ref1_led)
    # =============================================================================
    
    # =============================================================================
    # print(np.repeat(np.arange(kpos),krot*led_num),'haha')
    # print(np.tile(np.repeat(np.arange(krot),led_num),kpos),'haha')
    # print(np.tile(np.arange(led_num),kpos*krot),'haha')
    # print(ref1_led.flatten(),'haha')
    # =============================================================================
    # mask: 遮掉ref1
    maskled = np.full(light_led.shape, False)
    maskled[\
            np.repeat(np.arange(kpos),krot*led_num),\
            np.tile(np.repeat(np.arange(krot),led_num),kpos),\
            np.tile(np.arange(led_num),kpos*krot),\
            ref1_led.flatten()] = True #kp,kr,ledu, pd
    maskpd = np.full(light_pd.shape, False)
    maskpd[np.repeat(np.arange(kpos),krot*pd_num),\
        np.tile(np.repeat(np.arange(krot),pd_num),kpos),\
        ref1_pd.flatten(),
        np.tile(np.arange(pd_num),kpos*krot),\
        ] = True #kp,kr,led, pdu
    led_data_ref = light_led.copy()
    led_data_ref .mask = (light_led .mask | ~maskled)
    led_data_ref = np.sort(led_data_ref,axis=3)[:,:,:,0].reshape(kpos,krot,led_num,1)
    led_data_other = light_led.copy()
    led_data_other.mask = (light_led.mask | maskled)
    
    # led_data_ref = light_led[maskled].reshape(kpos,krot,-1,1)#kp kr ledu 1
    #led_data_other = light_led[~maskled].reshape(ledu,-1)# ledu other
    pd_data_ref = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
    pd_data_ref.mask = (light_pd.mask | ~maskpd)
    pd_data_ref = np.sort(pd_data_ref,axis=2)[:,:,0,:].reshape(kpos,krot,1,pd_num)
    pd_data_other = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
    pd_data_other.mask = (light_pd .mask | maskpd)
    # =============================================================================
    # print(light_pd,'-----------')
    # print(maskpd)
    # =============================================================================
    #pd_data_other = light_pd[~maskpd].reshape(led_num-1,-1) #other, pdu
    # ref/other
    #ratio_led = np.power(np.ma.divide(led_data_ref, led_data_other),1/pd_m) #led_u x other
    ratio_led = np.power(np.divide(led_data_ref, led_data_other),1/pd_m)
    ratio_pd = np.power(np.divide(pd_data_ref, pd_data_other),1/led_m) #other, pdu
    # in_ang  krot,kpos,led_num,pd_num
    
    check_inang_ref = (np.sort(np.ma.masked_array(in_ang_view,(light_led.mask | ~maskled)),axis=3)[:,:,:,0].reshape(kpos,krot,-1,1))
    check_inang_other = (np.ma.masked_array(in_ang_view,led_data_other.mask))
    check_ratio = np.divide(np.cos(check_inang_ref),np.cos(check_inang_other))
    check_ratio_sum = np.sum(~np.isclose(ratio_led,check_ratio,equal_nan=True))
    # print('-------------------------------------')
    # print('False Ratio:',check_ratio_sum)
    
    # =============================================================================
    # 計算平面normal vector[ledu other 3]
    # =============================================================================
    #kpos x krot x ledu x other x 3
    conf_led = np.tile(pd_ori_car.T,(kpos,krot,led_num,1,1))
    conf_led_ref = np.sort( (np.ma.masked_array(conf_led,np.tile((light_pd.mask | ~maskled),(3,1,1,1,1)).transpose(1,2,3,4,0))),axis=3)[:,:,:,0,:].reshape(kpos,krot,led_num,1,3)
    conf_led_other = np.ma.masked_array(conf_led,np.tile(led_data_other.mask,(3,1,1,1,1)).transpose(1,2,3,4,0))
    nor_led = conf_led_ref - np.multiply(ratio_led.reshape(kpos,krot,led_num,-1,1),conf_led_other)
    
    check_dot_led = np.sum(np.multiply(np.tile(testp_pos,(krot,led_num,pd_num,1,1)).transpose(4,0,1,2,3),nor_led),axis=4)
    check_dot_led = np.ma.masked_invalid(check_dot_led)
    #check_dot_led = np.sum(~(np.isclose(check_dot_led,np.zeros(led_data_other.shape))|np.isnan(check_dot_led)))
    check_dot_led_sum = np.sum(~(np.isclose(check_dot_led,np.zeros(check_dot_led.shape))))
    #print(check_dot_led_sum)
    
    
    # kp kr l p 3
    conf_pd = np.tile(led_ori_car,(kpos,krot,pd_num,1,1)).transpose(0,1,4,2,3) # kp kr l p
    # kp kr 1 p 3
    conf_pd_ref = np.sort( (np.ma.masked_array(conf_pd,np.tile((light_pd.mask | ~maskpd),(3,1,1,1,1)).transpose(1,2,3,4,0))),axis=2)[:,:,0,:,:].reshape(kpos,krot,1,-1,3)
    # kp kr l p 3
    conf_pd_other = np.ma.masked_array(conf_pd,np.tile(pd_data_other.mask,(3,1,1,1,1)).transpose(1,2,3,4,0))
    # kp kr l p
    nor_pd = conf_pd_ref - np.multiply(ratio_pd.reshape(kpos,krot,led_num,-1,1),conf_pd_other)
    # 驗算dot
    check_dot_pd = np.sum(np.multiply(glob_inv_pd_pos.reshape(kpos,krot,1,-1,3),nor_pd),axis=4)
    check_dot_pd = np.ma.masked_invalid(check_dot_pd)
    check_dot_pd_sum = np.sum(~(np.isclose(check_dot_pd,np.zeros(check_dot_pd.shape))))
    #print(check_dot_led_sum)
    # print('-------------------------------------')
    # print('False normal vector from pd view:' ,check_dot_led_sum)
    # print('False normal vector from led view:' ,check_dot_pd_sum)
    
    
    
    # =============================================================================
    # 取led_data_other強度最大者作為ref2_led，當cross的基準
    # 並利用maskled2將data other分兩半
    # => 計算cross
    # =============================================================================
    ref2_led = np.nanargmax(led_data_other, axis = 3)
    ref2_pd = np.nanargmax(pd_data_other, axis = 2) #pdu,
    
    
    maskled2 = np.full(light_led.shape, False)
    maskled2[\
            np.repeat(np.arange(kpos),krot*led_num),\
            np.tile(np.repeat(np.arange(krot),led_num),kpos),\
            np.tile(np.arange(led_num),kpos*krot),\
            ref2_led.flatten()] = True #kp,kr,ledu, pd
    
    maskpd2 = np.full(light_pd.shape, False)
    maskpd2[np.repeat(np.arange(kpos),krot*pd_num),\
        np.tile(np.repeat(np.arange(krot),pd_num),kpos),\
        ref2_pd.flatten(),
        np.tile(np.arange(pd_num),kpos*krot),\
        ] = True #kp,kr,led, pdu
    
    # 將normal vector分兩半
    
    nor_led_ref = nor_led.copy()
    nor_led_ref.mask = np.tile((light_led.mask| ~maskled2) ,(3,1,1,1,1)).transpose(1,2,3,4,0)#light_led True遮掉unusable, ~maskled2 True是除了ref2以外的 遮掉不是ref2的
    nor_led_ref = np.sort(nor_led_ref,axis=3)[:,:,:,0,:].reshape(kpos,krot,led_num,1,3)
    nor_led_other = nor_led.copy()
    nor_led_other.mask = (nor_led.mask|np.tile(maskled2,(3,1,1,1,1)).transpose(1,2,3,4,0))#nor_led True遮掉unusable、ref1, maskled2 True遮掉ref2的
    
    
    nor_pd_ref = nor_pd.copy()
    nor_pd_ref.mask = np.tile((light_pd.mask| ~maskpd2) ,(3,1,1,1,1)).transpose(1,2,3,4,0)#light_led True遮掉unusable, ~maskled2 True是除了ref2以外的 遮掉不是ref2的
    nor_pd_ref = np.sort(nor_pd_ref,axis=2)[:,:,0,:,:].reshape(kpos,krot,1,-1,3)
    nor_pd_other = nor_pd.copy()
    nor_pd_other.mask = (nor_pd.mask|np.tile(maskpd2,(3,1,1,1,1)).transpose(1,2,3,4,0))#nor_led True遮掉unusable、ref1, maskled2 True遮掉ref2的
    # print(nor_pd_other)
    # nor_pd_ref = nor_pd[maskpd2].reshape(1,-1,3) #1,pdu,3
    # nor_pd_other = nor_pd[~maskpd2].reshape(-1,pdu,3) #led-2,pdu,3
    
    # =============================================================================
    # # 計算各平面交軸：cross vector
    # =============================================================================
    # kp kr l p 3
    cross_led = np.ma.masked_array(np.cross(np.tile(nor_led_ref,(1,1,1,pd_num,1)),nor_led_other)\
                                   ,nor_led_other.mask )#ledu,other-1,3
    cross_led = np.divide(cross_led, np.tile(np.sqrt(np.sum(np.square(cross_led),axis=4)),(1,1,1,1,1)).transpose(1,2,3,4,0))#ledu,other-1,3
    cross_led_mask = np.sum(np.multiply(conf_led_ref, cross_led),axis=4)<0 ## kp kr l p
    cross_led = np.ma.masked_array(np.where(np.tile(cross_led_mask,(3,1,1,1,1)).transpose(1,2,3,4,0),-cross_led,cross_led),\
                                   nor_led_other.mask)
    # 驗算cross
    check_cross_led = np.sum(np.multiply(cross_led,(np.tile( \
                                                            np.divide(testp_pos,\
                                                                      np.tile(np.sqrt(np.sum(np.square(testp_pos),axis=0)),(1,1))\
                                                                      ),\
                                                            (krot,led_num,pd_num,1,1)).transpose(4,0,1,2,3))\
                                         ),axis=4)#kp kr l p
    check_cross_led = np.isclose(np.ma.masked_invalid(check_cross_led),np.ones(check_cross_led.shape))
    check_cross_led_sum = np.sum(~check_cross_led)
    cross_led_av = np.nanmean(cross_led,(2,3)) #kp kr 3
    
    # kp kr l p 3
    cross_pd = np.ma.masked_array(np.cross(np.tile(nor_pd_ref,(1,1,led_num,1,1)),nor_pd_other)\
                                   ,nor_pd_other.mask )#ledu,other-1,3
    
    cross_pd = np.divide(cross_pd, np.tile(np.sqrt(np.sum(np.square(cross_pd),axis=4)),(1,1,1,1,1)).transpose(1,2,3,4,0))#ledu,other-1,3
    
    cross_pd_mask = np.sum(np.multiply(conf_pd_ref, cross_pd),axis=4)<0 ## kp kr l p
    cross_pd = np.ma.masked_array(np.where(np.tile(cross_pd_mask,(3,1,1,1,1)).transpose(1,2,3,4,0),-cross_pd,cross_pd),\
                                   nor_pd_other.mask)
    
    # 驗算cross
    check_cross_pd = np.sum(np.multiply(cross_pd,(np.tile( \
                                                            np.divide(glob_inv_pd_pos,\
                                                                      np.tile(np.sqrt(np.sum(np.square(glob_inv_pd_pos),axis=3)),(1,1,1,1)).transpose(1,2,3,0)\
                                                                      ),\
                                                            (led_num,1,1,1,1)).transpose(1,2,0,3,4))\
                                         ),axis=4)#kp kr l p
    # print(check_cross_pd.transpose(0,1,3,2,4))
    check_cross_pd = np.isclose(np.ma.masked_invalid(check_cross_pd),np.ones(check_cross_pd.shape))
    check_cross_pd_sum = np.sum(~check_cross_pd)
    
    # print('------------------------------------')
    # print('False cross vector from pd view:' ,check_cross_led_sum)
    # print('False cross vector from pd view:' ,check_cross_pd_sum)
    
    
    
    # 答案求平均（忽略nan）
    ori_sol_pd_coor = np.nanmean(cross_led,axis = (2,3)).filled(fill_value=np.nan) #kp kr 3,
    ori_sol_led_coor = np.nanmean(cross_pd,axis = (2,3)).filled(fill_value=np.nan) #kp kr 3,
    # print(ori_sol_pd_coor)
    
    
    # 由答案算in_ang,out_ang - ori_sol 3,  - ori_pd 3,pd
    sol_in_ang = np.arccos(np.inner(ori_sol_pd_coor,pd_ori_car.T)) # kp kr pd,
    sol_out_ang = np.arccos(np.inner(ori_sol_led_coor,led_ori_car.T)) #kp kr led,
    # print(sol_out_ang)
    sol_dis = np.sqrt(const * np.divide(np.multiply(\
                                                    np.tile(np.power(np.cos(sol_in_ang),pd_m),(led_num,1,1,1)).transpose(1,2,0,3),\
                                                    np.tile(np.power(np.cos(sol_out_ang),led_m),(pd_num,1,1,1)).transpose(1,2,3,0)\
                                                    ),\
                                        light_f)) #kp kr l p
    # print(np.tile(sol_in_ang,(led_num,1,1,1)).transpose(1,2,0,3),';;;;;;;')
    # print(np.tile(sol_out_ang,(pd_num,1,1,1)).transpose(1,2,3,0),'---------')
    # print(sol_dis)
    check_dis = np.sqrt(np.sum(np.square(np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose(1,2,3,0,4)),axis=4))
    check_dis = np.sum(~np.isclose(np.ma.masked_invalid(sol_dis),check_dis))
    # print('------------------------------------')
    # print('False dis:' ,check_dis)
    # print('------------------------------------')
    sol_dis_av = np.nanmean(sol_dis,axis = (2,3))#kp kr
    # print()
    error = (np.sum(np.square(np.multiply(cross_led_av,sol_dis_av.reshape(kpos,-1,1))-glob_led_pos[:,:,0,:]),axis=2))
    unsolve = np.ma.count_masked(error)
    error = error.filled(np.inf) # masked改成inf
    error[error==0] = np.nan # sqrt不能處理0
    error = np.sqrt(error)
    error[np.isnan(error)]= 0
    
    tolerance = 0.05
    success = np.sum(error<tolerance)
    error_av = np.mean(error[error<tolerance])
    print(unsolve,success)
    # kp kr
    # print(ledu)
    # print(pdu)
    #print(error[0,0])
    return glob_led_pos,glob_led_ori,error ,unsolve,success,error_av


# initiate
testp_pos = np.array([[0,1,1],[0,0,1],[0,-1,2]]).T # 3x?
# kpos = testp_pos.shape[1]
testp_rot = np.array([[np.pi,0,0],[0,0,0]]).T
# krot = testp_rot.shape[1]
pd_num = 7
led_num = 5
led_m = 3
pd_m = 3



axis_color = 'lightgoldenrodyellow'

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(121,projection='3d')
ax.set_box_aspect(aspect = (1,1,1))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
ax.set_xlim3d(-1.5,1.5)
ax.set_ylim3d(-1.5,1.5)
ax.set_zlim3d(0,3)


# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

t = np.arange(0.0, 1.0, 0.001)
amp_0 = 5
freq_0 = 3

x,y,z = 0 ,0,1
a,b,c = np.pi,0,0

# bubble = ax.scatter(testp_pos[:,:,0],testp_pos[:,:,1],testp_pos[:,:,2],size = error+0.1)
# draw sphere
# =============================================================================
# u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,20))
# x = 0.1*np.cos(u)*np.sin(v)
# y = 0.1*np.sin(u)*np.sin(v)
# z = 0.1*np.cos(v)
# sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# =============================================================================

arrow = 0.3*np.array([[0,0,1]]).T
ax.quiver(0,0,0,arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=0.5, color='k')
arrow_rot = np.tile((testp_rot_matlist(testp_rot) @ arrow).squeeze(),(testp_pos.shape[1],1,1)) #kr 3 1
arrow_p = np.tile(testp_pos,(testp_rot.shape[1],1,1)).transpose(2,0,1)
# axis_item = ax.quiver(arrow_p[:,:,0],arrow_p[:,:,1],arrow_p[:,:,2],arrow_rot[:,:,0],arrow_rot[:,:,1],arrow_rot[:,:,2],arrow_length_ratio=0.5, color=["r"])


glob_led_pos,glob_led_ori,error,unsolve,success,error_av  = solve_mulmul(led_num,pd_num,led_m,pd_m)
text_item = ax.text(-2.5,-2.5,-2, f'Unsolvable:{unsolve}\nSuccess:{success}\nMean error:{error_av:.4E}')



error = error.flatten()
bubble = []
for i in range(error.size):
    if error[i] != np.inf:
        bubble.append(ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = 10**9*error+10,c = 'b'))
    else:
        bubble.append(ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],marker = 'x',c = 'k',s = 100))

# bubble = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = error+100)
#ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
# =============================================================================
# if type(glob_led_pos)!=type(None):
#     ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
# else: ans = ax.text2D(-0.14,-0.16,'No Answer',transform=ax.transAxes,color='k')
# =============================================================================
#text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
#print(vec,dis)
# =============================================================================
# text_item = ax.text(-2,-2,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}')
# =============================================================================

# Draw the initial plot
# The 'line' variable is used for modifying the line later
# =============================================================================
# [line] = ax.scatter(0,sol_in_ang, linewidth=2, color='red')
# ax.set_xlim([0, 1])
# ax.set_ylim([-10, 10])
# =============================================================================

# Add two sliders for tweaking the parameters
text = ['led amount','pd amount','led m','pd m']
init_val = np.array([led_num,pd_num,led_m,pd_m])
min_val = [3,3,2,2]
max_val = [20,20,70,70]
sliders = []
for i in np.arange(len(min_val)):

    axamp = plt.axes([0.84, 0.8-(i*0.05), 0.12, 0.02])
    # Slider
    # s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
    
    s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i],valstep=1)
    # else:
    #     s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
    sliders.append(s)


# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):

    global  sphere,axis_item,ans
    #ax.collections.remove(sphere)
    # ax.collections.remove(axis_item)
    #ax.collections.remove(ans)
    
    # arrow_rot = rotate_mat(np.array([sliders[3].val,sliders[4].val,sliders[5].val])) @ arrow
    # sphere = ax.plot_wireframe(x+sliders[0].val, y+sliders[1].val, z+sliders[2].val, color="w",alpha=0.2, edgecolor="#808080")   
    #axis_item = ax.quiver(sliders[0].val,sliders[1].val,sliders[2].val,arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=[0.5], color=["r"])
    glob_led_pos,glob_led_ori,error,unsolve,success,error_av = solve_mulmul(\
                    sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val)
    # arrow_rot = np.tile((testp_rot_matlist(testp_rot) @ arrow).squeeze(),(testp_pos.shape[1],1,1)) #kr 3 1
    # arrow_p = np.tile(testp_pos,(testp_rot.shape[1],1,1)).transpose(2,0,1)
    # axis_item = ax.quiver(arrow_p[:,:,0],arrow_p[:,:,1],arrow_p[:,:,2],arrow_rot[:,:,0],arrow_rot[:,:,1],arrow_rot[:,:,2],arrow_length_ratio=0.5, color=["r"])

        #text_item.set_text(f'Usable LED:{ledu} \nUsable PD:{pdu}')
    error = error.flatten()
    for i in range(error.size):
        ax.collections.remove(bubble[i])
        text_item.set_text(f'Unsolvable:{unsolve}\nSuccess:{success}\nMean error:{error_av:.4E}')
        if error[i] != np.inf:
            bubble[i] = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = 10**9*error+10,c = 'b',)
        else:
            bubble[i] = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],marker = 'x',c = 'k',s = 100)
# =============================================================================
#     if type(error)!=type(None):
#         bubble = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = error+000)
#         #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
#     else: 
#         bubble = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = 100,marker='x',c = 'k')
#         print('gg')
# =============================================================================
        #ans = ax.scatter(0,0,0,marker='x',color='k',s=10000)
    #text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
    #ax.text2D(0,0,'No Answer',transform=ax.transAxes)
                            # ax.collections.remove(arrow)
    fig.canvas.draw_idle()


for i in np.arange(len(min_val)):
    #samp.on_changed(update_slider)
    sliders[i].on_changed(sliders_on_changed)

# =============================================================================
# # Add a button for resetting the parameters
# reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
# reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
# def reset_button_on_clicked(mouse_event):
#     freq_slider.reset()
#     amp_slider.reset()
# reset_button.on_clicked(reset_button_on_clicked)
# =============================================================================


plt.show()