#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:56:27 2022

@author: tiffany
"""
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.cm as cm
import matplotlib.colors as colors

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

def solve_mulmul():
# set environment

    global threshold #= 0.001
    
    global pd_num #= int(pd_num)
    global led_num# = int(led_num)
    global pd_m #= int(pd_m)
    global led_m #= int(led_m)
    

    pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    #led_num = 5
    #led_m = 10
    led_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    # led_alpha = np.deg2rad(45)#傾角
    # led_beta = np.deg2rad(360/led_num)#方位角
    
    global pd_area #= 1
    global led_pt #= 1
    global pd_saturate #= np.inf
    global pd_respon# = 1
    
    # config
    pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
    global pd_ori_ang #= np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
    global pd_ori_car #= ori_ang2cart(pd_ori_ang) #3xpd
    pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3
    
    led_pos = np.tile(np.array([[0,0,0]]).T,(1,led_num))
    global led_ori_ang #= np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
    global led_ori_car #= ori_ang2cart(led_ori_ang) #3xled
    led_rot_mat = rotate_z_mul(led_ori_ang[1,:]) @ rotate_y_mul(led_ori_ang[0,:])#ledx3x3
    
    
    # sample point
    global testp_pos# = (np.mgrid[-1:1:4j, -1:1:4j, 1:3:4j].reshape(-1,4*4*4)) # 3x?
    # kpos = testp_pos.shape[1]
    global testp_rot #= np.array([[np.pi,0,0],[0,np.pi,0]]).T
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
    # print(in_ang,'in')
    # print(out_ang,'out')
    
    # 在view angle外的寫nan
    
    in_ang_view = filter_view_angle(in_ang,pd_view)
    out_ang_view = filter_view_angle(out_ang,led_view)
    # in_ang_view[np.cos(in_ang_view)<0]=np.nan
    # out_ang_view[np.cos(out_ang_view)<0]=np.nan
    in_ang_view[in_ang_view>=np.pi/2]=np.nan
    out_ang_view[out_ang_view>=np.pi/2]=np.nan
    # print(out_ang_view,'out')
    
    const = pd_respon * pd_area * led_pt * (led_num+1)/(2*np.pi)
    light = const * np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
    # light = np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
    mask_light= np.isnan(light)
    # print(np.sum(light<0),'here')
    # light[mask_light] = 0
    
    # print(light)
    # =============================================================================
    # 這裡處理加上noise的部分
    # =============================================================================
    
    boltz = 1.380649 * 10**(-23)
    temp_k = 300
    elec_charge = 1.60217663 * 10**(-19)
    
    global bandwidth #= 300
    global shunt #= 50
    global back_ground,dark_current,NEP

    thermal_noise = 4*temp_k*boltz*bandwidth/shunt
    
    noise = 1*np.sqrt(thermal_noise\
              + 2*elec_charge*bandwidth*(light+background+dark_current)\
                  ) #+ 2*elec_charge*bandwidth*dark
    # print(noise[0,0,0,0])
    # print(shunt)
    # print(noise)
    light_noise = light + noise
    light_floor = NEP*np.floor_divide(light_noise, NEP)
    
    
    # print(np.nanmax(light_floor),'!!!!!')
    # -------以下是硬體部分------------------
    
    
    # snr = np.divide(noise,light_floor)
    # print(snr)
    
    # filter掉訊號中小於threshold的部分：nan
    # krot,kpos,led_num,pd_num
    light_f = np.copy(light_floor)
    
    light_f[light_f <= threshold] = np.nan
    light_f[light_f >= pd_saturate] = np.nan
    # print(light)
    # print(light_f<=threshold)
    
    
    
    
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
    global ledu,pdu
    ledu = led_usable.sum(axis=2)#kp,kr
    pdu = pd_usable.sum(axis=2)#kp,kr
    # print(ledu,pdu)
    

    nor_led,nor_pd,conf_led_ref,conf_pd_ref,led_data_other,pd_data_other = get_surface(light_led,light_pd,led_num,pd_num,kpos,krot,led_m,pd_m,led_ori_car,pd_ori_car)
    cross_led,cross_pd = get_cross(led_data_other,pd_data_other,light_led,light_pd,led_num,pd_num,kpos,krot,nor_led,nor_pd,conf_led_ref,conf_pd_ref)
    
    # weight_form = 'mean''weight'
    
    global weight_form 
    mask_led = ~np.isnan(cross_led[:,:,:,:,0].filled(fill_value=np.nan))
    mask_pd = ~np.isnan(cross_pd[:,:,:,:,0].filled(fill_value=np.nan))
    mask_total = (mask_led|mask_pd)# kp kr l p 3
    mask_count = np.sum(mask_total,axis=(2,3)).reshape((kpos,krot,1,1))
    if weight_form =='mean':
        # weight = np.nansum(~mask_total[:,:,:,:,0],axis=(2,3)).reshape(kpos,krot,1,1) 
        weight = np.divide(mask_total,mask_count)
    elif weight_form =='weight':
        weight = np.ma.masked_array(light_f,mask_total[:,:,:,:,0])
        weight = np.power(weight,3)
        weight_sum = np.nansum(weight,axis=(2,3)).reshape((kpos,krot,1,1))
        weight = np.divide(weight,weight_sum)
    
    # 答案求平均（忽略nan）
    
    
    
    # weight = np.nansum(np.power(light_f,1/3),axis=(2,3)).reshape(kpos,krot,1,1) # kp kr
    # weight = np.divide(np.power(light_f,1/3),weight)
    # check = np.nansum(weight,axis=(2,3))
    # print(light_f[3,3,:,:])
    # print(check)
    
    # cross_led_av = np.multiply(cross_led,weight)
    # ori_sol_pd_coor = np.sum(np.multiply(cross_led,np.tile(weight,(3,1,1,1,1)).transpose((1,2,3,4,0))),axis=(2,3)).filled(fill_value=np.nan)
    # ori_sol_led_coor = np.sum(np.multiply(cross_pd,np.tile(weight,(3,1,1,1,1)).transpose((1,2,3,4,0))),axis=(2,3)).filled(fill_value=np.nan)
    global ori_sol_pd_coor,ori_sol_led_coor
    ori_sol_pd_coor = np.nanmean(cross_led,axis = (2,3))#.filled(fill_value=np.nan) #kp kr 3,
    ori_sol_led_coor = np.nanmean(cross_pd,axis = (2,3))#.filled(fill_value=np.nan) #kp kr 3,
    # print(ori_sol_pd_coor.shape,ori_sol_led_coor.shape)
    
    
    # 由答案算in_ang,out_ang - ori_sol 3,  - ori_pd 3,pd
    sol_in_ang = np.arccos(np.inner(ori_sol_pd_coor,pd_ori_car.T)) # kp kr pd,
    sol_out_ang = np.arccos(np.inner(ori_sol_led_coor,led_ori_car.T)) #kp kr led,
    # print(sol_out_ang)
    sol_dis = np.sqrt(const * np.divide(np.multiply(\
                                                    np.tile(np.power(np.cos(sol_in_ang),pd_m),(led_num,1,1,1)).transpose(1,2,0,3),\
                                                    np.tile(np.power(np.cos(sol_out_ang),led_m),(pd_num,1,1,1)).transpose(1,2,3,0)\
                                                    ),\
                                        light_f)) #kp kr l p
        
    # check_dis = np.sqrt(np.sum(np.square(np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose(1,2,3,0,4)),axis=4))
    # check_dis = np.sum(~np.isclose(np.ma.masked_invalid(sol_dis),check_dis))
    # print('------------------------------------')
    # print('False dis:' ,check_dis)
    # print('------------------------------------')
    global sol_dis_av
    # sol_dis_av= np.sum(np.multiply(sol_dis,weight),axis=(2,3))
    # print(sol_dis_av.shape,'~~~')
    sol_dis_av = np.nanmean(sol_dis,axis = (2,3))#kp kr
    # print()
    global error
    error = (np.sum(np.square(np.multiply(ori_sol_pd_coor,sol_dis_av.reshape(kpos,-1,1))-glob_led_pos[:,:,0,:]),axis=2))
    global  unsolve
    global  solve
    solve = np.ma.count(error)
    unsolve = np.ma.count_masked(error)
    error = error.filled(np.inf) # masked改成inf
    error[error==0] = np.nan # sqrt不能處理0
    error = np.sqrt(error)
    error[np.isnan(error)]= 0
    

    
    global tolerance 
    global success
    success = np.sum(error<tolerance)
    global error_av 
    error_av = np.mean(error[error<tolerance])
    # print('unsolve:',unsolve,', solve:',solve,', success:',success)
    # print(np.average(sol_dis_av))
    
    # return glob_led_pos,glob_led_ori,error ,unsolve,success,error_av

def set_hardware(led_hard,pd_hard):
    led_list = [\
                1.7*np.pi,80*10**(-3)
                ]
    # respon area NEP darkcurrent shunt
    pd_list = [\
               [0.64, 6*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9],\
               [0.64, 5.7*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9],\
               [0.64, 33*10**(-6), 2*10**(-15), 50*10**(-12), 10*10**9],\
               [0.64, 100*10**(-6), 2.8*10**(-15), 200*10**(-12), 5*10**9],\
               [0.38, 36*10**(-6), 3.5*10**(-14), 100*10**(-12), 0.1*10**9]\
               ]
    pd_list = np.array(pd_list)
    # print(pd_list[pd_hard,:])
    # print(led_list[led_hard])
    return led_list[led_hard],pd_list[pd_hard,:]

def set_scenario(scenario):
    if scenario ==0:
        testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 0:3:10j].reshape((3,-1)) # 3x?
        # testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
        testp_rot  = np.deg2rad(np.mgrid[0:0:1j, 10:60:6j, 36:360:10j].reshape((3,-1)))
        testp_rot = np.concatenate((testp_rot,np.array([[0,0,0]]).T ),axis=1)+np.array([[np.pi,0,0]]).T
    elif scenario ==1:
        testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 3:3:10j].reshape((3,-1)) # 3x?
        testp_rot = np.array([[np.pi,0,0]]).T
    elif scenario ==2:
        sample = 6
        dis_sample = np.linspace(0,3,4+1)[1:]
        # testp_rot  = np.deg2rad(np.mgrid[0:0:1j, 10:60:6j, 36:360:10j].reshape((3,-1)))
        u, v = np.meshgrid(np.linspace(0,2*np.pi,2*sample+1)[0:-1:1],np.linspace(0,np.pi,sample+1)[1:-1:1])
        u = np.append(u.reshape((-1,)),0)
        v = np.append(v.reshape((-1,)),0)
        x = (1*np.cos(u)*np.sin(v))
        y = (1*np.sin(u)*np.sin(v))
        z =( 1*np.cos(v))
        U = np.stack((x,y,z))
        print(U.shape)
        U = np.tile(U,(dis_sample.size,1,1)).transpose((1,2,0))
        testp_pos = np.multiply(dis_sample.reshape((1,1,-1)),U)
        # testp_pos = np.concatenate((U,2*U,3*U),axis = 0)
        testp_pos = testp_pos.reshape((3,-1))
        # testp_pos = 3*U
        print(testp_pos.shape[1],'kpos')
        
        testp_rot = np.stack((np.zeros(u.shape),v,u))
        # testp_rot = np.concatenate((testp_rot,np.array([[]])))
    return testp_pos,testp_rot

def set_config(config_num,led_alpha,pd_alpha):
    if config_num ==0:
        # pd_alpha = np.deg2rad(10)
        # led_alpha = np.deg2rad(10)
        
        pd_beta = np.deg2rad(360/pd_num)#方位角
        pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
        pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
    
        led_beta = np.deg2rad(360/led_num)#方位角
        led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
        led_ori_car = ori_ang2cart(led_ori_ang) #3xled
    return led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car

# =============================================================================
# # 前置
# =============================================================================

# 從演算法裡面copy出來的有用資訊
sol_dis_av=[]
unsolve = 0
success = 0
error = []
error_av = []
ledu = 0
pdu = 0
ori_sol_pd_coor = []
ori_sol_led_coor = []

# =============================================================================
# # set environment
# =======================================================================
threshold = 10**(-7)
tolerance = 0.1
effective = 80
weight_form = 'mean'

# 硬體參數

led_hard = 1
pd_hard = 3
led_para,pd_para = set_hardware(led_hard, pd_hard)
led_pt = led_para
pd_respon,pd_area,NEP,dark_current,shunt = pd_para

background = 740*10**(-6)
pd_saturate = 10#np.inf


bandwidth = 9.7**13/10 # 320-1000nm
bandwidth = 370*10**3
# shunt = 1000*10**6 # 10-1000 mega

# mode = 'scenario'
# mode = 'analysis'
# mode = 'interactive_1to1'
mode = 'interactive_mulmul'
scenario = 0
config_num = 0




if mode =='scenario':
    testp_pos,testp_rot = set_scenario(scenario)
    # testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 0:3:10j].reshape((3,-1)) # 3x?
    # testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
    # testp_rot  = np.deg2rad(np.mgrid[0:0:1j, 10:60:6j, 36:360:10j].reshape((3,-1)))
    # testp_rot = np.concatenate((testp_rot,np.array([[0,0,0]]).T ),axis=1)+np.array([[np.pi,0,0]]).T

    kpos = testp_pos.shape[1]
    krot = testp_rot.shape[1]

    # solve_mulmul()
    # count_kpos = np.nansum(error<tolerance,axis=1)/krot
    # count_krot = np.nansum(error<tolerance,axis=0)/kpos

    fig = plt.figure(figsize=(12, 8))
    # colormap= plt.cm.get_cmap('YlOrRd')
    # normalize =  colors.Normalize(vmin=0, vmax=1)

    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.set_box_aspect(aspect = (1,1,1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(True)
    ax.set_title('平移樣本點')
    if scenario ==2:
        
        ax.set_xlim3d(-3,3)
        ax.set_ylim3d(-3,3)
        ax.set_zlim3d(-3,3)
    else:
        ax.set_xlim3d(-1.5,1.5)
        ax.set_ylim3d(-1.5,1.5)
        ax.set_zlim3d(0,3)

    sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],alpha=0.5)
    ax.scatter(0,0,0,color='k',marker='x')

    # fig.colorbar(sc,shrink=0.3,pad=0.1)

    ax = fig.add_subplot(1,3,3,projection='polar')

    sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  )
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    ax.set_title('以極座標系呈現Pitch,Yaw')

    ax.grid(True)

    ax = fig.add_subplot(1,3,2,projection='3d')
    ax.set_title('旋轉樣本點')
    ax.set_box_aspect(aspect = (1,1,1))
    ax.grid(False)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_axis_off()

    # ax.scatter(0,0,0,color='k',marker='x')

    u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,20))
    x = 1*np.cos(u)*np.sin(v)
    y = 1*np.sin(u)*np.sin(v)
    z = 1*np.cos(v)
    # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
    ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

    a = np.linspace(-1,1,21)
    b = np.linspace(-1,1,21)
    A,B = np.meshgrid(a,b)
    c = np.zeros((21,21))
    ax.plot_surface(A,B,c, color="grey",alpha=0.2)

    a,b,c1 = ori_ang2cart(testp_rot[1:,:])

    ax.quiver(0,0,0,0,0,-1,color='r')
    ax.quiver(0,0,0,a,b,c1,color = 'b')


elif mode=='analysis':
    pd_num = 8
    led_num = 8
    led_m = 1
    pd_m = 1
    
    # ans = np.zeros((14,14,5,5,5,5,2))
    # numl = np.array([3,5,8,10,12,15])
    # # numl = np.arange(3,16,1)
    # nump = np.arange(8,9,1)
    # ml = np.array([1,1.5,2,3,5])
    # mp = np.array([1,1.5,2,3,5])
    # alphal = np.deg2rad(np.arange(10,60,10))
    # alphap = np.deg2rad(np.arange(10,60,10))
    
    pd_alpha = np.deg2rad(10)#傾角
    pd_beta = np.deg2rad(360/pd_num)#方位角
    pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
    pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
    
    led_alpha = np.deg2rad(10)#傾角
    led_beta = np.deg2rad(360/led_num)#方位角
    led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
    led_ori_car = ori_ang2cart(led_ori_ang) #3xled
    
    
    # =============================================================================
    # # set sample points
    # =============================================================================
    # testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 0:3:10j].reshape((3,-1)) # 3x?
    # testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
    # testp_rot  = np.deg2rad(np.mgrid[0:0:1j, 10:60:6j, 36:360:10j].reshape((3,-1)))
    # testp_rot = np.concatenate((testp_rot,np.array([[0,0,0]]).T ),axis=1)+np.array([[np.pi,0,0]]).T
    testp_pos,testp_rot = set_scenario(scenario)
    kpos = testp_pos.shape[1]
    krot = testp_rot.shape[1]
    
    solve_mulmul()
    count_kpos = np.nansum(error<tolerance,axis=1)
    count_krot = np.nansum(error<tolerance,axis=0)
    
    fig = plt.figure(figsize=(12, 8))
    colormap= plt.cm.get_cmap('YlOrRd')
    normalizep =  colors.Normalize(vmin=0, vmax=krot)
    normalizer =  colors.Normalize(vmin=0, vmax=kpos)
    fig.subplots_adjust(wspace=0.3)
    
    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.set_box_aspect(aspect = (1,1,1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(True)
    if scenario ==2:
        ax.set_xlim3d(-3,3)
        ax.set_ylim3d(-3,3)
        ax.set_zlim3d(-3,3)
    else:
        ax.set_xlim3d(-1.5,1.5)
        ax.set_ylim3d(-1.5,1.5)
        ax.set_zlim3d(0,3)
    
    sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap=colormap,norm = normalizep,alpha=0.5)
    ax.scatter(0,0,0,color='k',marker='x')
    
    colorbar = fig.colorbar(sc,shrink=0.3,pad=0.15)
    ax.set_title('平移樣本點')
    
    colorbar.ax.set_ylabel('容許範圍內的樣本點數量')
    
    
    
    ax = fig.add_subplot(1,3,2,projection='polar')
    
    sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
    colorbar = fig.colorbar(sc,shrink=0.3,pad=0.15)
    colorbar.ax.set_ylabel('容許範圍內的樣本點數量')
    ax.set_title('旋轉樣本點')
    ax.text(1,1,'pitch(degree)',rotation = 15)
    ax.text(np.deg2rad(60),80,'yaw(degree)')
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    
    
    ax.grid(True)
    

elif mode =='interactive_1to1':

    # initiate
    testp_pos = np.array([[0,1,1]]).T # 3x?
    #kpos = testp_pos.shape[1]
    testp_rot = np.array([[np.pi,0,0]]).T
    #krot = testp_rot.shape[1]
    pd_num = 7
    led_num = 7
    led_m = 3
    pd_m = 3
    
    pd_alpha = np.deg2rad(10)
    led_alpha = np.deg2rad(10)
    
    led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)


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


    # draw sphere
    u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,20))
    x = 0.1*np.cos(u)*np.sin(v)
    y = 0.1*np.sin(u)*np.sin(v)
    z = 0.1*np.cos(v)
    sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
    ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

    arrow = 0.5*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
    ax.quiver(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=[0.2,0.5], color='b')
    arrow_rot = rotate_mat(testp_rot) @ arrow
    axis_item = ax.quiver(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=0.1, color='b')


    solve_mulmul()
    pdu = pdu[0,0]
    ledu = ledu[0,0]
    error = error[0,0]
    dis = sol_dis_av[0,0]
    vec = ori_sol_pd_coor[0,0,:]

    #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
    if ledu==0|pdu==0:
        ans = ax.scatter(0,0,0,marker='x',color='k',s=10000)
        text_item = ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error}')
        error_vec =ax.quiver (0,0,0,1,1,1,alpha=0)
    else:
        ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
        text_item = ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error:.4E}')
        error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0]-dis*vec[0],testp_pos[1,0]-dis*vec[1],testp_pos[2,0]-dis*vec[2],color = 'r')
        # error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0,0],testp_pos[0,0,1],testp_pos[0,0,2],color = 'r')

        #text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
    #print(vec,dis)

    # Add two sliders for tweaking the parameters
    text = ['x','y','z','roll','pitch','yaw','led amount','pd amount','led m','pd m','shunt(Ohm)','bandwidth(Hz)','led_alpha(rad)','pd_alpha(rad)']
    init_val = np.append(np.concatenate((testp_pos,testp_rot)).flatten(),(led_num,pd_num,led_m,pd_m,shunt,bandwidth,led_alpha,pd_alpha))
    min_val = [-1.5,-1.5,0,0,0,0,3,3,2,2,10**6,10**3,0,0]
    max_val = [1.5,1.5,3,2*np.pi,2*np.pi,2*np.pi,20,20,70,70,10**9,10**12,np.pi,np.pi]
    sliders = []
    for i in np.arange(len(min_val)):

        axamp = plt.axes([0.74, 0.8-(i*0.05), 0.12, 0.02])
        # Slider
        # s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
        if 8>i >5:
            s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i],valstep=1)
        else:
            s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
        sliders.append(s)


    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):

        global  sphere,axis_item,ans,error_vec
        ax.collections.remove(sphere)
        ax.collections.remove(axis_item)
        ax.collections.remove(ans)
        ax.collections.remove(error_vec)
        #ax.collections.remove(text_num)
        
        global testp_pos,testp_rot
        testp_pos = np.array([[sliders[0].val,sliders[1].val,sliders[2].val]]).T
        testp_rot = np.array([[sliders[3].val,sliders[4].val,sliders[5].val]]).T
        arrow_rot = rotate_mat(np.array([sliders[3].val,sliders[4].val,sliders[5].val])) @ arrow
        sphere = ax.plot_wireframe(x+sliders[0].val, y+sliders[1].val, z+sliders[2].val, color="w",alpha=0.2, edgecolor="#808080")   
        axis_item = ax.quiver(sliders[0].val,sliders[1].val,sliders[2].val,arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=[0.2,0.5], color='b')
        
        global pd_num,led_num,pd_m,led_m,pd_alpha,led_alpha,error,ledu,pdu,bandwidth,shunt,led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car
        led_num = int(sliders[6].val)
        pd_num = int(sliders[7].val)
        led_m = sliders[8].val
        pd_m = sliders[9].val
        shunt = sliders[10].val
        bandwidth = sliders[11].val
        led_alpha = sliders[12].val
        pd_alpha = sliders[13].val
        led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)

        
        solve_mulmul()
        
        
        pdu = pdu[0,0]
        ledu = ledu[0,0]
        error = error[0,0]
        dis = sol_dis_av[0,0]
        vec = ori_sol_pd_coor[0,0,:]
    
        #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
        if ledu==0|pdu==0:
            ans = ax.scatter(0,0,0,marker='x',color='k',s=10000)
            text_item.set_text(f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error}')
            error_vec =ax.quiver (0,0,0,1,1,1,alpha=0)
        else:
            ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
            text_item .set_text(f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error:.4E}')
            error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0]-dis*vec[0],testp_pos[1,0]-dis*vec[1],testp_pos[2,0]-dis*vec[2],color = 'r')
            # error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0,0],testp_pos[0,0,1],testp_pos[0,0,2],color = 'r')

        fig.canvas.draw_idle()


    for i in np.arange(len(min_val)):
        #samp.on_changed(update_slider)
        sliders[i].on_changed(sliders_on_changed)

elif mode =='interactive_mulmul':

    # initiate
    testp_pos ,testp_rot = set_scenario(scenario)
    kpos = testp_pos.shape[1]
    # testp_rot = np.array([[np.pi,0,0]]).T
    krot = testp_rot.shape[1]
    
    pd_num = 3
    led_num = 3
    led_m = 3
    pd_m = 3
    
    pd_alpha = np.deg2rad(10)
    led_alpha = np.deg2rad(10)
    
    led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
    
    solve_mulmul()
    count_kpos = np.nansum(error<tolerance,axis=1)
    count_krot = np.nansum(error<tolerance,axis=0)
    effective_pos = count_kpos/krot >=effective/100
    effective_rot = count_krot/kpos >=effective/100

    fig = plt.figure(figsize=(12, 8))
    colormap= plt.cm.get_cmap('YlOrRd')
    normalizep =  colors.Normalize(vmin=0, vmax=krot)
    normalizer =  colors.Normalize(vmin=0, vmax=kpos)
    fig.subplots_adjust(wspace=0.3)
    
    ax1 = fig.add_subplot(2,3,1,projection='3d')
    ax1.set_box_aspect(aspect = (1,1,1))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.grid(True)
    if scenario ==2:
        ax1.set_xlim3d(-3,3)
        ax1.set_ylim3d(-3,3)
        ax1.set_zlim3d(-3,3)
    else:
        ax1.set_xlim3d(-1.5,1.5)
        ax1.set_ylim3d(-1.5,1.5)
        ax1.set_zlim3d(0,3)
    
    sc1 = ax1.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap=colormap,norm = normalizep,alpha=0.5)
    ax1.scatter(0,0,0,color='k',marker='x')
    
    colorbar = fig.colorbar(sc1,shrink=0.3,pad=0.15)
    ax1.set_title('平移樣本點')
    
    colorbar.ax.set_ylabel('容許範圍內的樣本點數量')
    
    
    
    ax2 = fig.add_subplot(2,3,4,projection='polar')
    
    sc2 = ax2.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
    colorbar = fig.colorbar(sc2,shrink=0.3,pad=0.15)
    colorbar.ax.set_ylabel('容許範圍內的樣本點數量')
    ax2.set_title('旋轉樣本點')
    ax2.text(1,1,'pitch(degree)',rotation = 15)
    ax2.text(np.deg2rad(60),80,'yaw(degree)')
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    
    ax3 = fig.add_subplot(2,3,2,projection='3d')
    ax3.set_box_aspect(aspect = (1,1,1))
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.grid(True)
    if scenario ==2:
        ax3.set_xlim3d(-3,3)
        ax3.set_ylim3d(-3,3)
        ax3.set_zlim3d(-3,3)
    else:
        ax3.set_xlim3d(-1.5,1.5)
        ax3.set_ylim3d(-1.5,1.5)
        ax3.set_zlim3d(0,3)
    
    sc3 = ax3.scatter(testp_pos[0,effective_pos],testp_pos[1,effective_pos],testp_pos[2,effective_pos],color = 'b',alpha=0.5)
    ax3.scatter(0,0,0,color='k',marker='x')

    ax3.set_title('平移樣本有效範圍\n')
    
    
    
    
    ax4 = fig.add_subplot(2,3,5,projection='polar')
    
    sc4 = ax4.scatter(testp_rot[2,effective_rot],np.rad2deg(testp_rot[1,effective_rot])  ,color = 'b')

    ax4.set_title('旋轉樣本有效範圍')
    ax4.text(1,1,'pitch(degree)',rotation = 15)
    ax4.text(np.deg2rad(60),80,'yaw(degree)')
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    
    
    #
        #text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
    #print(vec,dis)

    # Add two sliders for tweaking the parameters
    text = ['tolerance(m)','effective(%)','led amount','pd amount','led m','pd m','shunt(Ohm)','bandwidth(Hz)','led_alpha(rad)','pd_alpha(rad)','pd_saturate']
    init_val = np.array((tolerance,effective,led_num,pd_num,led_m,pd_m,shunt,bandwidth,led_alpha,pd_alpha,pd_saturate))
    min_val = [0,0,3,3,2,2,10**6,10**3,0,0,10*(-3)]
    max_val = [1,100,20,20,70,70,10**9,10**12,np.pi,np.pi,10]
    sliders = []
    for i in np.arange(len(min_val)):

        axamp = plt.axes([0.74, 0.8-(i*0.05), 0.12, 0.02])
        # Slider
        # s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
        if 4>i >1:
            s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i],valstep=1)
        else:
            s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
        sliders.append(s)


    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):

        global  sc1,sc2,sc3,sc4,ax1,ax2,ax3,ax4
        ax1.collections.remove(sc1)
        ax2.collections.remove(sc2)
        ax3.collections.remove(sc3)
        ax4.collections.remove(sc4)
        
        global tolerance,effective,pd_saturate
        global pd_num,led_num,pd_m,led_m,pd_alpha,led_alpha,error,ledu,pdu,bandwidth,shunt,led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car
        tolerance = sliders[0].val
        effective = sliders[1].val
        led_num = int(sliders[2].val)
        pd_num = int(sliders[3].val)
        led_m = sliders[4].val
        pd_m = sliders[5].val
        shunt = sliders[6].val
        bandwidth = sliders[7].val
        led_alpha = sliders[8].val
        pd_alpha = sliders[9].val
        pd_saturate = sliders[10].val
        led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)

        
        solve_mulmul()
        solve_mulmul()
        count_kpos = np.nansum(error<tolerance,axis=1)
        count_krot = np.nansum(error<tolerance,axis=0)
        effective_pos = count_kpos/krot >=effective/100
        effective_rot = count_krot/kpos >=effective/100
        
        sc1 = ax1.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap=colormap,norm = normalizep,alpha=0.5)
        sc2 = ax2.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
        sc3 = ax3.scatter(testp_pos[0,effective_pos],testp_pos[1,effective_pos],testp_pos[2,effective_pos],color = 'b',alpha=0.5)
        sc4 = ax4.scatter(testp_rot[2,effective_rot],np.rad2deg(testp_rot[1,effective_rot])  ,color = 'b')

    
        #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
        
        fig.canvas.draw_idle()


    for i in np.arange(len(min_val)):
        #samp.on_changed(update_slider)
        sliders[i].on_changed(sliders_on_changed)


# ans = np.zeros((18,18,2))
# fig1 = plt.figure(figsize=(15, 15))
# fig2 = plt.figure(figsize=(15, 15))
# cm = plt.cm.get_cmap('rainbow')

# for numli in range(numl.size):
#     for numpi in range(nump.size):
        
#         # pd_alpha = alphap[alphapi]
#         # led_alpha = alphal[alphali]
#         # pd_num = nump[numpi]
#         # led_num = numl[numli]
#         led_num = numl[numli]
#         pd_num = nump[numpi]
        
#         pd_beta = np.deg2rad(360/pd_num)#方位角
#         pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
#         pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd

#         led_beta = np.deg2rad(360/led_num)#方位角
#         led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
#         led_ori_car = ori_ang2cart(led_ori_ang) #3xled
        
#         solve_mulmul()
#         count_kpos = np.sum((error<tolerance),axis=1)/testp_rot.shape[1]
#         count_krot = np.sum((error<tolerance),axis=0)/testp_pos.shape[1]
        
        
#         ans[numli,numpi,0] = success
#         ans[numli,numpi,1] = error_av
        
#         # cm = plt.cm.get_cmap('rainbow')

#         # ax = fig1.add_subplot(5,5,1+mli*5+mpi,projection='3d')
#         # ax.set_box_aspect(aspect = (1,1,1))
#         # ax.set_xlabel('x')
#         # ax.set_ylabel('y')
#         # ax.set_zlabel('z')
#         # ax.grid(True)
#         # ax.set_xlim3d(-1.5,1.5)
#         # ax.set_ylim3d(-1.5,1.5)
#         # ax.set_zlim3d(0,1.5)

#         # sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap='rainbow')
#         # ax.scatter(0,0,0,color='k',marker='x')

#         # fig1.colorbar(sc,shrink=0.3,pad=0.1)

#         # ax = fig2.add_subplot(5,5,1+mli*5+mpi,projection='polar')
#         # # ax.axis('equal')
        
#         # sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap='rainbow')
        
        
#         # fig2.colorbar(sc,shrink=0.3,pad=0.1)
        
#         # ax.grid(True)
        
#         # # ax = fig.add_subplot(1,3,2,projection='3d')
#         # # ax.set_box_aspect(aspect = (1,1,0.5))
#         # # ax.grid(False)
#         # # ax.set_xlim3d(-1,1)
#         # # ax.set_ylim3d(-1,1)
#         # # ax.set_zlim3d(0,1)
#         # # ax.xaxis.set_ticklabels([])
#         # # ax.yaxis.set_ticklabels([])
#         # # ax.zaxis.set_ticklabels([])
#         # # ax.set_axis_off()

#         # # ax.scatter(0,0,0,color='k',marker='x')

#         # # u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,20))
#         # # x = 1*np.cos(u)*np.sin(v)
#         # # y = 1*np.sin(u)*np.sin(v)
#         # # z = 1*np.cos(v)
#         # # # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
#         # # ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

#         # # a = np.linspace(-1,1,21)
#         # # b = np.linspace(-1,1,21)
#         # # A,B = np.meshgrid(a,b)
#         # # c = np.zeros((21,21))
#         # # ax.plot_surface(A,B,c, color="grey",alpha=0.2)

#         # # a,b,c1 = ori_ang2cart(testp_rot[1:,:])

#         # # ax.quiver(0,0,0,0,0,-1,color='r')
#         # # ax.quiver(0,0,0,a,b,c1,color = 'b')

#         # ax = fig.add_subplot(1,3,2,projection='polar')
#         # # ax.axis('equal')



#         # sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap='rainbow')


#         # fig.colorbar(sc,shrink=0.3,pad=0.1)

#         # ax.grid(True)

#         # fig.suptitle((f'L={led_num},P={pd_num},Ml={led_m},Mp={pd_m}'))

# c = 0 

# for numli in range(numl.size):
#     for numpi in range(nump.size):
#         for mli in range(ml.size):
#             for mpi in range(mp.size):
#                 for alphali in range(alphal.size):
#                     for alphapi in range(alphap.size):
#                         pd_alpha = alphap[alphapi]
#                         led_alpha = alphal[alphali]
#                         pd_num = nump[numpi]
#                         led_num = numl[numli]
#                         led_m = ml[mli]
#                         pd_m = mp[mpi]
                        
#                         pd_beta = np.deg2rad(360/pd_num)#方位角
#                         pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
#                         pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd

#                         led_beta = np.deg2rad(360/led_num)#方位角
#                         led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
#                         led_ori_car = ori_ang2cart(led_ori_ang) #3xled
                        
#                         solve_mulmul()
#                         ans[numli,numpi,mli,mpi,alphali,alphapi,0] = success
#                         ans[numli,numpi,mli,mpi,alphali,alphapi,1] = error_av
#                         c = c+1
#                         print(c)







# =============================================================================
# 
# solve_mulmul()
# 
# 
# count_kpos = np.sum((error<tolerance),axis=1)/testp_rot.shape[1]
# count_krot = np.sum((error<tolerance),axis=0)/testp_pos.shape[1]
# 
# 
# 
# 
# 
# # plot sample points
# axis_color = 'lightgoldenrodyellow'
# 
# # colors = cm.rainbow()
# 
# fig = plt.figure(figsize=(12, 8))
# 
# cm = plt.cm.get_cmap('rainbow')
# 
# ax = fig.add_subplot(1,3,1,projection='3d')
# ax.set_box_aspect(aspect = (1,1,1))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.grid(True)
# ax.set_xlim3d(-1.5,1.5)
# ax.set_ylim3d(-1.5,1.5)
# ax.set_zlim3d(0,3)
# 
# sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap='rainbow',alpha=0.5)
# ax.scatter(0,0,0,color='k',marker='x')
# 
# fig.colorbar(sc,shrink=0.3,pad=0.1)
# 
# =============================================================================

# ax = fig.add_subplot(1,3,2,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()

# ax.scatter(0,0,0,color='k',marker='x')

# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,20))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)

# a,b,c1 = ori_ang2cart(testp_rot[1:,:])

# ax.quiver(0,0,0,0,0,-1,color='r')
# ax.quiver(0,0,0,a,b,c1,color = 'b')

# =============================================================================
# ax = fig.add_subplot(1,3,2,projection='polar')
# # ax.axis('equal')
# 
# 
# 
# sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap='rainbow')
# 
# 
# fig.colorbar(sc,shrink=0.3,pad=0.1)
# 
# ax.grid(True)
# 
# fig.suptitle((f'L={led_num},P={pd_num},Ml={led_m},Mp={pd_m}'))
# =============================================================================




# =============================================================================
# # =============================================================================
# # plot
# # =============================================================================
# 
# axis_color = 'lightgoldenrodyellow'
# 
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1,2,1,projection='3d')
# ax.set_box_aspect(aspect = (1,1,1))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.grid(True)
# ax.set_xlim3d(-1.5,1.5)
# ax.set_ylim3d(-1.5,1.5)
# ax.set_zlim3d(0,3)
# 
# 
# # Adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.25, bottom=0.25)
# 
# 
# 
# 
# # bubble = ax.scatter(testp_pos[:,:,0],testp_pos[:,:,1],testp_pos[:,:,2],size = error+0.1)
# # draw sphere
# # =============================================================================
# # u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,20))
# # x = 0.1*np.cos(u)*np.sin(v)
# # y = 0.1*np.sin(u)*np.sin(v)
# # z = 0.1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# # ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# # =============================================================================
# 
# arrow = 0.3*np.array([[0,0,1]]).T
# ax.quiver(0,0,0,arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=0.5, color='k')
# arrow_rot = np.tile((testp_rot_matlist(testp_rot) @ arrow).squeeze(),(testp_pos.shape[1],1,1)) #kr 3 1
# arrow_p = np.tile(testp_pos,(testp_rot.shape[1],1,1)).transpose(2,0,1)
# # axis_item = ax.quiver(arrow_p[:,:,0],arrow_p[:,:,1],arrow_p[:,:,2],arrow_rot[:,:,0],arrow_rot[:,:,1],arrow_rot[:,:,2],arrow_length_ratio=0.5, color=["r"])
# 
# 
# glob_led_pos,glob_led_ori,error,unsolve,success,error_av  = solve_mulmul(led_num,pd_num,led_m,pd_m)
# text_item = ax.text(-2.5,-2.5,-2, f'Unsolvable:{unsolve}\nSuccess:{success}\nMean error:{error_av:.4E}')
# 
# 
# 
# error = error.flatten()
# bubble = []
# for i in range(error.size):
#     if error[i] != np.inf:
#         bubble.append(ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = 10**9*error+10,c = 'b'))
#     else:
#         bubble.append(ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],marker = 'x',c = 'k',s = 100))
# 
# 
# # Add two sliders for tweaking the parameters
# text = ['led amount','pd amount','led m','pd m']
# init_val = np.array([led_num,pd_num,led_m,pd_m])
# min_val = [3,3,2,2]
# max_val = [20,20,70,70]
# sliders = []
# 
# 
# for i in np.arange(len(min_val)):
# 
#     axamp = plt.axes([0.84, 0.8-(i*0.05), 0.12, 0.02])
#     # Slider
#     # s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
#     
#     s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i],valstep=1)
#     # else:
#     #     s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
#     sliders.append(s)
# 
# 
# # Define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 
#     global  sphere,axis_item,ans
#     #ax.collections.remove(sphere)
#     led_num,pd_num,led_m,pd_m = sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val
#     
#     pd_alpha = np.deg2rad(35)#傾角
#     pd_beta = np.deg2rad(360/pd_num)#方位角
#     global pd_ori_ang 
#     pd_ori_ang= np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
#     global pd_ori_car 
#     pd_ori_car= ori_ang2cart(pd_ori_ang) #3xpd
# 
#     led_alpha = np.deg2rad(45)#傾角
#     led_beta = np.deg2rad(360/led_num)#方位角
#     global led_ori_ang 
#     led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
#     global led_ori_car 
#     led_ori_car= ori_ang2cart(led_ori_ang) #3xled
#     
#     glob_led_pos,glob_led_ori,error,unsolve,success,error_av = solve_mulmul(led_num,pd_num,led_m,pd_m )
# 
#     error = error.flatten()
#     for i in range(error.size):
#         ax.collections.remove(bubble[i])
#         text_item.set_text(f'Unsolvable:{unsolve}\nSuccess:{success}\nMean error:{error_av:.4E}')
#         if error[i] != np.inf:
#             bubble[i] = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = 10**9*error+10,c = 'b',)
#         else:
#             bubble[i] = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],marker = 'x',c = 'k',s = 100)
# 
#     fig.canvas.draw_idle()
# 
# 
# for i in np.arange(len(min_val)):
#     #samp.on_changed(update_slider)
#     sliders[i].on_changed(sliders_on_changed)
# 
# 
# 
# plt.show()
# =============================================================================
