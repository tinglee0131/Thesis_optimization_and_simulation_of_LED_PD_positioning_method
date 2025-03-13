#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from turtle import back
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
    # pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3
    
    led_pos = np.tile(np.array([[0,0,0]]).T,(1,led_num))
    global led_ori_ang #= np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
    global led_ori_car #= ori_ang2cart(led_ori_ang) #3xled
    # led_rot_mat = rotate_z_mul(led_ori_ang[1,:]) @ rotate_y_mul(led_ori_ang[0,:])#ledx3x3
    
    
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
    # mask_light= np.isnan(light)
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
    global background,dark_current,NEP,capacitance

    shunt = 1/(2*np.pi*bandwidth*capacitance)

    thermal_noise = 4*temp_k*boltz*bandwidth/shunt
    
    noise_var = 1*np.sqrt(thermal_noise\
              + 2*elec_charge*bandwidth*(light+background+dark_current)\
                  ) #+ 2*elec_charge*bandwidth*dark
    # print(noise_var,'var')
    np.random.seed(10)
    noise = np.random.standard_normal(size = light.shape)
    noise = np.multiply(noise,noise_var)
    # print(noise,'noise')
    # print(np.nanmax(noise),'noise')
    # print(noise[0,0,0,0])
    # print(shunt)
    # print(noise)
    global gain
    light_noise = gain*(light) + noise
    # NEP = NEP# *np.sqrt(bandwidth)
    
    light_floor = light_noise # NEP*np.floor_divide(light_noise, NEP)
    # print(np.nanmean(light),'light')
    
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
    
    # print(light_f,'light')
    
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
    
    # print(light,'light')
    nor_led,nor_pd,conf_led_ref,conf_pd_ref,led_data_other,pd_data_other = get_surface(light_led,light_pd,led_num,pd_num,kpos,krot,led_m,pd_m,led_ori_car,pd_ori_car)
    cross_led,cross_pd = get_cross(led_data_other,pd_data_other,light_led,light_pd,led_num,pd_num,kpos,krot,nor_led,nor_pd,conf_led_ref,conf_pd_ref)
    
    # print('=============')
    # print(nor_led,'norled')
    # print(nor_pd,'norpd')
    # print(cross_led,'crossled')
    # print(cross_pd,'crosspd')
    # weight_form = 'mean''weight'
    
    global weight_form 
    mask_led = ~np.isnan(cross_led[:,:,:,:,0].filled(fill_value=np.nan))
    mask_pd = ~np.isnan(cross_pd[:,:,:,:,0].filled(fill_value=np.nan))
    mask_total = (mask_led|mask_pd)# kp kr l p 3
    mask_count = np.sum(mask_total,axis=(2,3)).reshape((kpos,krot,1,1))

    
    # 答案求平均（忽略nan）
    global ori_sol_pd_coor,ori_sol_led_coor
    if weight_form =='weight':
        # global ori_sol_pd_coor,ori_sol_led_coor
    # kp kr l p 
        mask_led = np.isnan((cross_led[:,:,:,:,0]).filled(fill_value=np.nan))#True是不要的
        weight_led = np.ma.array(light,mask = mask_led).filled(fill_value=np.nan)
        total_led = np.nansum(weight_led,axis=(2,3))
        total_led = np.tile(total_led,(1,1,1,1)).transpose(2,3,0,1)# kp kr 1 1
        weight_led = np.divide(weight_led,total_led) # kp kr l p 
        weight_led = np.tile(weight_led,(1,1,1,1,1)).transpose(1,2,3,4,0)
        sol_led = np.multiply(weight_led,cross_led)
        ori_sol_pd_coor = np.nansum(sol_led,axis=(2,3))
        
        mask_pd = np.isnan((cross_pd[:,:,:,:,0]).filled(fill_value=np.nan))#True是不要的
        weight_pd = np.ma.array(light,mask = mask_pd).filled(fill_value=np.nan)
        total_pd = np.nansum(weight_pd,axis=(2,3))
        total_pd = np.tile(total_pd,(1,1,1,1)).transpose(2,3,0,1)# kp kr 1 1
        weight_pd = np.divide(weight_pd,total_pd) # kp kr l p 
        weight_pd = np.tile(weight_pd,(1,1,1,1,1)).transpose(1,2,3,4,0)# kp kr l p 3
        sol_pd = np.multiply(weight_pd,cross_pd)
        ori_sol_led_coor = np.nansum(sol_pd,axis=(2,3)) # kp kr 3
        
    # cross_led_av = np.multiply(cross_led,weight)
    # ori_sol_pd_coor = np.sum(np.multiply(cross_led,np.tile(weight,(3,1,1,1,1)).transpose((1,2,3,4,0))),axis=(2,3)).filled(fill_value=np.nan)
    # ori_sol_led_coor = np.sum(np.multiply(cross_pd,np.tile(weight,(3,1,1,1,1)).transpose((1,2,3,4,0))),axis=(2,3)).filled(fill_value=np.nan)
    elif weight_form =='mean':
        
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



def set_hardware(led_hard,pd_hard):
    led_list = [\
                1.7*np.pi, 80*10**(-3), 1.35,1.15
                ]
    # respon area NEP darkcurrent shunt
    pd_list = [\
                # pd_respon,pd_area,NEP,dark_current,shunt,capacitance
               [0.64, 6*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9, 740*10**(-12)],\
               [0.64, 5.7*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9, 680*10**(-12)],\
               [0.64, 33*10**(-6), 2*10**(-15), 50*10**(-12), 10*10**9, 4000*10**(-12)],\
               [0.64, 100*10**(-6), 2.8*10**(-15), 200*10**(-12), 5*10**9, 13000*10**(-12)],\
               [0.38, 36*10**(-6), 3.5*10**(-14), 100*10**(-12), 0.1*10**9, 700*10**(-12)]\
               ]
    pd_list = np.array(pd_list)
    # print(pd_list[pd_hard,:])
    # print(led_list[led_hard])
    return led_list[led_hard],pd_list[pd_hard,:]

def set_scenario(scenario):
    if scenario ==0:
        testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 0:3:10j].reshape((3,-1)) # 3x?
        # testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
        testp_rot  = ((np.mgrid[0:0:1j, 10:60:6j, 0:360:11j])[:,:,:,:-1])
        # print(testp_rot,testp_rot.shape)
        testp_rot = np.deg2rad(testp_rot.reshape((3,-1)))
        testp_rot = np.concatenate((testp_rot,np.array([[0,0,0]]).T ),axis=1)+np.array([[np.pi,0,0]]).T
    
    elif scenario ==1:
        testp_pos = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j, 2.5:2.5:1j].reshape((3,-1)) # 3x?
        testp_rot = np.array([[np.pi,0,0]]).T
        print(testp_pos[0,:].shape)
    elif scenario ==2:
        sample = 6
        dis_sample = np.linspace(0,3,4+1)[1:]
        # testp_rot  = np.deg2rad(np.mgrid[0:0:1j, 10:60:6j, 36:360:10j].reshape((3,-1)))
        u, v = np.meshgrid(np.linspace(0,2*np.pi,2*sample+1)[0:-1:1],np.linspace(0,np.pi,sample+1)[1:-1:1])
        print(u.shape,v.shape)
        u = np.append(u.reshape((-1,)),0)
        v = np.append(v.reshape((-1,)),0)
        u = np.append(u.reshape((-1,)),0)
        v = np.append(v.reshape((-1,)),np.pi)
        x = (1*np.cos(u)*np.sin(v))
        y = (1*np.sin(u)*np.sin(v))
        z =( 1*np.cos(v))
        U = np.stack((x,y,z))
        # print(U.shape)
        U = np.tile(U,(dis_sample.size,1,1)).transpose((1,2,0))
        testp_pos = np.multiply(dis_sample.reshape((1,1,-1)),U)
        # testp_pos = np.concatenate((U,2*U,3*U),axis = 0)
        testp_pos = testp_pos.reshape((3,-1))
        # testp_pos = 3*U
        print(testp_pos.shape[1],'kpos')
        
        
        testp_rot = np.stack((np.zeros(u.shape),v,u))
        print(testp_rot.shape[1],'krot')
        # testp_rot = np.concatenate((testp_rot,np.array([[]])))
    if scenario ==3:
        global ma
        testp_pos = np.mgrid[-ma/2:ma/2:10j, -ma/2:ma/2:10j, 0:ma:10j].reshape((3,-1)) # 3x?
        # testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
        testp_rot  = ((np.mgrid[0:0:1j, 10:60:6j, 0:360:11j])[:,:,:,:-1])
        # print(testp_rot,testp_rot.shape)
        testp_rot = np.deg2rad(testp_rot.reshape((3,-1)))
        testp_rot = np.concatenate((testp_rot,np.array([[0,0,0]]).T ),axis=1)+np.array([[np.pi,0,0]]).T
    
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
    elif config_num ==1:
        # pd_alpha = np.deg2rad(10)
        # led_alpha = np.deg2rad(10)
        
        pd_beta = np.deg2rad(360/(pd_num-1))#方位角
        pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num-1),(pd_beta*np.arange(1,pd_num))),0 )#2x?
        pd_ori_ang = np.concatenate((pd_ori_ang,np.array([[0,0]]).T),axis=1)
        pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
    
        led_beta = np.deg2rad(360/(led_num-1))#方位角
        led_ori_ang = np.stack( (led_alpha*np.ones(led_num-1),(led_beta*np.arange(1,led_num))),0 )#2x?
        led_ori_ang = np.concatenate((led_ori_ang,np.array([[0,0]]).T),axis=1)
        led_ori_car = ori_ang2cart(led_ori_ang) #3xled
    elif config_num==2:
        pd_a = (pd_num*0.4)//1
        pd_beta = np.deg2rad(360/pd_a)#方位角
        pd_ori_anga = np.stack( (pd_alpha*np.ones(int(pd_a)),(pd_beta*np.arange(1,pd_a+1))))#2x?
        pd_a = pd_num-pd_a
        pd_beta = np.deg2rad(360/pd_a)#方位角
        pd_ori_angb = np.stack( (3*pd_alpha*np.ones(int(pd_a),),(pd_beta*np.arange(1,pd_a+1))),0 )#2x?
        pd_ori_ang = np.concatenate((pd_ori_anga,pd_ori_angb),axis=1)
        pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
        
        led_a = (led_num*0.4)//1
        led_beta = np.deg2rad(360/led_a)#方位角
        led_ori_anga = np.stack( (led_alpha*np.ones(int(led_a)),(led_beta*np.arange(1,led_a+1))),0 )#2x?
        led_a = led_num-led_a
        led_beta = np.deg2rad(360/led_a)#方位角
        led_ori_angb = np.stack( (3*led_alpha*np.ones(int(led_a)),(led_beta*np.arange(1,led_a+1))),0 )#2x?
        led_ori_ang = np.concatenate((led_ori_anga,led_ori_angb),axis=1)
        led_ori_car = ori_ang2cart(led_ori_ang) #3xpd
        
        lista = np.deg2rad(np.array((-20,10,110)))
        mat = rotate_mat( lista )
        led_ori_car = mat @ led_ori_car
        
        lista = np.deg2rad(np.array((50,30,70)))
        mat = rotate_mat( lista )
        pd_ori_car = mat @ pd_ori_car

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
threshold = 10**(-9)
tolerance = 0.05
effective = 80
weight_form = 'mean'
# weight_form = 'weight'

# 硬體參數

led_hard = 3
pd_hard = 2
led_para,pd_para = set_hardware(led_hard, pd_hard)
led_pt = led_para
pd_respon,pd_area,NEP,dark_current,shunt,capacitance = pd_para

background = 5100*10**(-6)#
# background = 740*10**(-6)
pd_saturate = 10*10**(-3)#np.inf

shunt = 10**3 # 10-1000 mega
# bandwidth = 9.7**13/10 # 320-1000nm
bandwidth = 370*10**3
# bandwidth = 10**3

mode = 'scenario'
# mode = 'analysis'
# mode = 'interactive_1to1'
mode = 'interactive_mulmul'
# mode = 'save'
# mode = 'analysis_graph'
mode = 'config_interactive'
# mode = 'effect_plot'
# mode = 'draw_config'


scenario = 2
config_num = 0
rot_max = 180#180
gain = 1.14
ma = 10



if mode == 'draw_config':

    pd_num = 10
    led_num =10
    config_num = 0
    led_alpha = np.deg2rad(50)
    pd_alpha = np.deg2rad(50)
    led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num,led_alpha,pd_alpha)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,3,1,projection='3d')
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

    # a,b,c1 = ori_ang2cart(testp_rot[1:,:])

    ax.quiver(0,0,0,0,0,1,color='k')
    ax.quiver(0,0,0,1,0,0,color='k')
    ax.quiver(0,0,0,0,1,0,color='k')

    circle =  np.stack((led_alpha* np.ones((100)),\
        (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
    circle_cart = ori_ang2cart(circle)
    ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color='g',alpha=0.5)
    ax.text(1.1,0,0,'x')
    ax.text(0,1.1,0,'y')
    ax.text(0,0,1.1,'z')

    zeros = np.zeros(led_ori_car[0,:].shape)
    ax.quiver(zeros,zeros,zeros,led_ori_car[0,:],led_ori_car[1,:],led_ori_car[2,:],color = 'g')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1))
    plt.show()

elif   mode =='scenario':
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
    ax.set_title('平移樣本相對位置')
    if scenario ==2:
        
        ax.set_xlim3d(-3,3)
        ax.set_ylim3d(-3,3)
        ax.set_zlim3d(-3,3)
    elif scenario ==3:
        ax.set_xlim3d(-ma/2,ma/2)
        ax.set_ylim3d(-ma/2,ma/2)
        ax.set_zlim3d(0,ma)
    else:
        ax.set_xlim3d(-1.5,1.5)
        ax.set_ylim3d(-1.5,1.5)
        ax.set_zlim3d(0,3)

    sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],alpha=0.5,color = 'b',label = 'LED座標系位置')
    ax.scatter(0,0,0,color='r',marker='x',label='PD座標系位置')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    # fig.colorbar(sc,shrink=0.3,pad=0.1)

    ax = fig.add_subplot(1,3,3,projection='polar')

    sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:]) ,color='b' )
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    ax.set_title('以極座標系呈現旋轉樣本的Pitch,Yaw')

    ax.grid(True)

    ax = fig.add_subplot(1,3,2,projection='3d')
    ax.set_title('旋轉樣本相對姿態')
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

    ax.quiver(0,0,0,0,0,-1,color='r',label= 'PD座標系Z軸')
    ax.quiver(0,0,0,a,b,c1,color = 'b',label= 'LED座標系Z軸')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1))
    plt.show()

elif mode == 'analysis':
    # print('hi')
    pd_num = 3
    led_num = 3
    led_m = 5.3
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
    plt.show()

elif mode =='interactive_1to1':

    # initiate
    testp_pos = np.array([[1,1,1.5]]).T # 3x? 1,1,2.5  -1,1,2
    #kpos = testp_pos.shape[1]
    testp_rot = np.array([[np.pi,1.833,0]]).T
    #krot = testp_rot.shape[1]
    pd_num = 5
    led_num = 5
    led_m = 1
    pd_m = 1
    
    pd_alpha = np.deg2rad(50)
    led_alpha = np.deg2rad(50)
    
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
    x = 0.5*np.cos(u)*np.sin(v)
    y = 0.5*np.sin(u)*np.sin(v)
    z = 0.5*np.cos(v)
    sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
    ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

    # zeros = np.zeros(pd_ori_car[0,:].shape)
    # ax.quiver(zeros,zeros,zeros,0.5*pd_ori_car[0,:],0.5*pd_ori_car[1,:],0.5*pd_ori_car[2,:],color = 'orange',alpha=0.8,label='PD硬體')
    # arrow_rot = rotate_mat(testp_rot) @ led_ori_car
    # axis_item = ax.quiver(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],0.5*arrow_rot[0,:],0.5*arrow_rot[1,:],0.5*arrow_rot[2,:],color = 'purple',alpha=0.8,label='LED硬體')
    # ax.text(testp_pos[0,0]-0.8,testp_pos[1,0],testp_pos[2,0],'z',color='b')
    # ax.text(testp_pos[0,0],testp_pos[1,0]-0.8,testp_pos[2,0],'y',color='b')
    # ax.text(testp_pos[0,0],testp_pos[1,0],testp_pos[2,0]-0.8,'x',color='b')
    
    arrow = 0.5*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
    ax.quiver(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=[0.2,0.5], color='firebrick',label='PD座標系')
    arrow_rot = rotate_mat(testp_rot) @ arrow
    axis_item = ax.quiver(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=0.1, color='b',label='LED座標系')
    # plt.quiverkey(axis_item,0,0,1, label='LED座標系', labelpos='N', labelcolor='b')
    
    led_text=[0,0,0]
    ax.text(0,0,0.5,'z',color='r')
    ax.text(0,0.5,0,'y',color='r')
    ax.text(0.5,0,0,'x',color='r')
    # print(arrow_rot[0,1],arrow_rot.shape,testp_pos[0,0],'!!!!!!!!!!!!!!!!!!!!!')
    led_text[0] = ax.text(testp_pos[0,0]+1.1*arrow_rot[0,0][0],testp_pos[1,0]+1.1*arrow_rot[1,0],testp_pos[2,0]+1.1*arrow_rot[2,0],'x',color = 'b')
    led_text[1] = ax.text(testp_pos[0,0]+1.1*arrow_rot[0,1][0],testp_pos[1,0]+1.1*arrow_rot[1,1],testp_pos[2,0]+1.1*arrow_rot[2,1],'y',color = 'b')
    led_text[2] = ax.text(testp_pos[0,0]+1.1*arrow_rot[0,2][0],testp_pos[1,0]+1.1*arrow_rot[1,2],testp_pos[2,0]+1.1*arrow_rot[2,2],'z',color = 'b')
    

    ax.quiver(0,0,0,0,0,0,color = 'k',label='計算出的相對位置')
    ax.quiver(0,0,0,0,0,0,color = 'magenta',label='誤差')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

    # ax.text([0.6,0,0],[0,0.6,0],[0,0,0.6],['x','y','z'],c=['r','r','r'])
    # led_text = ax.text(testp_pos[0,:]+arrow_rot[0,:],testp_pos[1,:]+arrow_rot[1,:],testp_pos[2,:]+arrow_rot[2,:],['x','y','z'],color = 'b')

    solve_mulmul()
    pdu = pdu[0,0]
    ledu = ledu[0,0]
    error = error[0,0]
    dis = sol_dis_av[0,0]
    vec = ori_sol_pd_coor[0,0,:]

    #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
    if ledu==0 or pdu==0:
        ans = ax.scatter(0,0,0,marker='x',color='k',s=10000)
        text_item = ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error}')
        error_vec =ax.quiver (0,0,0,1,1,1,alpha=0,color = 'magenta')
    else:
        ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
        text_item = ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error:.4E}')
        error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0]-dis*vec[0],testp_pos[1,0]-dis*vec[1],testp_pos[2,0]-dis*vec[2],color = 'magenta')
        # pass
        
        # error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0,0],testp_pos[0,0,1],testp_pos[0,0,2],color = 'r')

        #text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
    #print(vec,dis)

    bandwidth_log = np.log10( bandwidth)
    background_log = np.log10(background)
    pd_saturate_log = np.log10(pd_saturate)
    # Add two sliders for tweaking the parameters
    text = [r'$^{PL}t_x$',r'$^{PL}t_y$',r'$^{PL}t_z$',\
            r'$Roll ^{PL}rx$',r'$Pitch ^{PL}ry$',r'$Yaw ^{PL}rz$',\
            r'LED數量$L$',r'PD數量$P$',r'LED朗博次方$M\ell$',r'PD朗博次方$Mp$',\
            r'背景電流$Ib$(A)',r'頻寬$B$(Hz)',\
            r'LED指向天頂角$^L\alpha$(deg)',r'PD指向天頂角$^P\alpha$(deg)',\
            r'PD飽和電流$Ib$(A)',r'多重路徑增益$Gm$']
    # print(len(text))
    init_val = np.append(np.concatenate((testp_pos,testp_rot)).flatten(),(led_num,pd_num,led_m,pd_m,background,bandwidth_log,np.rad2deg(led_alpha),np.rad2deg(pd_alpha),pd_saturate_log,gain))
    # print(init_val.shape)
    min_val = [-1.5,-1.5,0,0,0,\
                0,3,3,1,1,\
                -6,3,0,0,-6,1]
    max_val = [1.5,1.5,3,2*np.pi,2*np.pi,\
                2*np.pi,20,20,10,10,\
                -1,12,180,180,1,2]
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
    sliders[11].valtext.set_text(f'{bandwidth:.4E}')
    sliders[10].valtext.set_text(f'{background:.4E}')
    sliders[14].valtext.set_text(f'{pd_saturate:.4E}')
    # sliders[10].valtext.set_text(f'{shunt:.4E}')

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):

        global  ax, sphere,axis_item,ans,error_vec,led_text
        ax.collections.remove(sphere)
        ax.collections.remove(axis_item)
        ax.collections.remove(ans)
        ax.collections.remove(error_vec)
        # ax.collections.remove(led_text[0])
        # ax.collections.remove(led_text[1])
        # ax.collections.remove(led_text[2])
        # print('hi')
        #ax.collections.remove(text_num)
        
        global testp_pos,testp_rot
        testp_pos = np.array([[sliders[0].val,sliders[1].val,sliders[2].val]]).T
        testp_rot = np.array([[sliders[3].val,sliders[4].val,sliders[5].val]]).T
        arrow_rot = rotate_mat(np.array([sliders[3].val,sliders[4].val,sliders[5].val])) @ arrow
        sphere = ax.plot_wireframe(x+sliders[0].val, y+sliders[1].val, z+sliders[2].val, color="w",alpha=0.2, edgecolor="#808080")   
        axis_item = ax.quiver(sliders[0].val,sliders[1].val,sliders[2].val,arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=[0.2,0.5], color='b')
        
        global pd_num,led_num,pd_m,led_m,pd_alpha,led_alpha,error,ledu,pdu,bandwidth,background,led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car,pd_saturate,gain
        led_num = int(sliders[6].val)
        pd_num = int(sliders[7].val)
        led_m = sliders[8].val
        pd_m = sliders[9].val
        background_log = sliders[10].val
        bandwidth_log = sliders[11].val
        led_alpha = np.deg2rad(sliders[12].val)
        pd_alpha = np.deg2rad(sliders[13].val)
        pd_saturate_log = sliders[14].val
        gain = sliders[15].val
        led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
        background = 10**background_log
        bandwidth = 10**bandwidth_log
        pd_saturate = 10**pd_saturate_log
        
        
        solve_mulmul()
        led_text[0].remove()
        led_text[1].remove()
        led_text[2].remove()
        
        led_text[0] = ax.text(testp_pos[0,0]+1.1*arrow_rot[0,0],testp_pos[1,0]+1.1*arrow_rot[1,0],testp_pos[2,0]+1.1*arrow_rot[2,0],'x',color = 'b')
        led_text[1] = ax.text(testp_pos[0,0]+1.1*arrow_rot[0,1],testp_pos[1,0]+1.1*arrow_rot[1,1],testp_pos[2,0]+1.1*arrow_rot[2,1],'y',color = 'b')
        led_text[2] = ax.text(testp_pos[0,0]+1.1*arrow_rot[0,2],testp_pos[1,0]+1.1*arrow_rot[1,2],testp_pos[2,0]+1.1*arrow_rot[2,2],'z',color = 'b')
    
        pdu = pdu[0,0]
        ledu = ledu[0,0]
        error = error[0,0]
        dis = sol_dis_av[0,0]
        vec = ori_sol_pd_coor[0,0,:]
    
        #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
        if ledu==0  or pdu==0:
            print(ledu,pdu)
            print('bye')
            ans = ax.scatter(0,0,0,marker='x',color='k',s=10000)
            text_item.set_text(f'Usable LED:{ledu} \nUsable PD:{pdu}\nError: -')
            error_vec =ax.quiver (0,0,0,1,1,1,alpha=0)
        else:
            print(ledu,pdu)
            print(ledu==0,pdu==0)
            # print(error)
            ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
            text_item .set_text(f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error:.4E}')
            error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0]-dis*vec[0],testp_pos[1,0]-dis*vec[1],testp_pos[2,0]-dis*vec[2],color = 'magenta')
            # error_vec = ax.quiver(dis*vec[0],dis*vec[1],dis*vec[2],testp_pos[0,0,0],testp_pos[0,0,1],testp_pos[0,0,2],color = 'r')
        sliders[11].valtext.set_text(f'{bandwidth:.4E}')
        sliders[10].valtext.set_text(f'{background:.4E}')
        sliders[14].valtext.set_text(f'{pd_saturate:.4E}')
        fig.canvas.draw_idle()

    # plt.show()


    for i in np.arange(len(min_val)):
        #samp.on_changed(update_slider)
        # print('he')
        sliders[i].on_changed(sliders_on_changed)

    plt.show()

elif mode =='interactive_mulmul':
    max_temp = 0
    max_led = []
    max_pd = []
    # initiate
    testp_pos ,testp_rot = set_scenario(scenario)
    kpos = testp_pos.shape[1]
    # testp_rot = np.array([[np.pi,0,0]]).T
    krot = testp_rot.shape[1]
    # effective = 80



    pd_num = 5
    led_num = 5
    led_m = 1#5.3
    pd_m = 1
    
    pd_alpha = np.deg2rad(50)
    led_alpha = np.deg2rad(50)
    
    led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
    
    solve_mulmul()
    count_total = np.nansum(error<tolerance)
    print(count_total)
    count_kpos = np.nansum(error<tolerance,axis=1)
    count_krot = np.nansum(error<tolerance,axis=0)
    effective_pos = count_kpos/krot >=effective/100
    effective_rot = count_krot/kpos >=effective/100

    fig = plt.figure(figsize=(15, 8))
    colormap= plt.cm.get_cmap('YlOrRd')
    normalizep =  colors.Normalize(vmin=0, vmax=krot)
    normalizer =  colors.Normalize(vmin=0, vmax=kpos)
    fig.subplots_adjust(wspace=0.3,hspace=0.3)
    
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
    elif scenario ==3:
        ax1.set_xlim3d(-ma/2,ma/2)
        ax1.set_ylim3d(-ma/2,ma/2)
        ax1.set_zlim3d(0,ma)
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
    
    sc2 = ax2.scatter((testp_rot[2,:]),np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
    colorbar = fig.colorbar(sc2,shrink=0.3,pad=0.15)
    colorbar.ax.set_ylabel('容許範圍內的樣本點數量')
    ax2.set_title('旋轉樣本點')
    ax2.text(1,1,'pitch(degree)',rotation = 15)
    ax2.text(np.deg2rad(60),80,'yaw(degree)')
    ax2.set_ylim([0,rot_max])
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
    
    sc4 = ax4.scatter((testp_rot[2,effective_rot]),np.rad2deg(testp_rot[1,effective_rot])  ,color = 'b')

    ax4.set_title('旋轉樣本有效範圍')
    ax4.text(0,0,'pitch(degree)',rotation = 15)
    ax4.text(np.deg2rad(60),80,'yaw(degree)')
    ax4.set_ylim([0,rot_max])
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    
    fig.suptitle(f'於容許範圍內的樣本點總數：{count_total}')
    
    max_temp=count_total
    max_led = led_ori_ang
    max_pd = pd_ori_ang
    
    
    max_text = fig.text(0.8,0.1,f'Max:{max_temp}')

    #
        #text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
    #print(vec,dis)

    # Add two sliders for tweaking the parameters
    bandwidth_log = np.log10( bandwidth)
    background_log = np.log10(background)
    pd_saturate_log = np.log10(pd_saturate)
    shunt_log = np.log10(shunt)
    # Add two sliders for tweaking the parameters
    text = [r'容許範圍$To$(m)',r'有效比例(%)',r'LED數量$L$',r'PD數量$P$',\
            r'LED朗博次方$M\ell$',r'PD朗博次方$Mp$',\
            r'背景電流$Ib$(A)',r'頻寬$B$(Hz)',\
            r'LED指向天頂角$^L\alpha$(deg)',r'PD指向天頂角$^P\alpha$(deg)',\
            r'PD飽和電流$Ib$(A)',r'電阻$Rl$(Ohm)','Gain']
    init_val = np.array((tolerance,effective,led_num,pd_num,led_m,pd_m,background_log,bandwidth_log,np.rad2deg(led_alpha),np.rad2deg(pd_alpha),pd_saturate_log,shunt_log,gain))
    min_val = [0,0,3,3,1,1,-6,3,0,0,(-6),3,1]
    max_val = [0.5,100,20,20,10,10,-3,12,180,180,1,10,3]
    sliders = []
    for i in np.arange(len(min_val)):

        axamp = plt.axes([0.76, 0.8-(i*0.05), 0.12, 0.02])
        # Slider
        # s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
        if 4>i >1:
            s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i],valstep=1)
        else:
            s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
        sliders.append(s)

    sliders[7].valtext.set_text(f'{bandwidth:.4E}')
    sliders[6].valtext.set_text(f'{background:.4E}')
    sliders[10].valtext.set_text(f'{pd_saturate:.4E}')
    sliders[11].valtext.set_text(f'{shunt:.4E}')


    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):

        global  sc1,sc2,sc3,sc4,ax1,ax2,ax3,ax4
        ax1.collections.remove(sc1)
        ax2.collections.remove(sc2)
        ax3.collections.remove(sc3)
        ax4.collections.remove(sc4)
        
        global tolerance,effective,pd_saturate,shunt,gain
        global pd_num,led_num,pd_m,led_m,pd_alpha,led_alpha,error,ledu,pdu,bandwidth,background,led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car
        tolerance = sliders[0].val
        effective = sliders[1].val
        led_num = int(sliders[2].val)
        pd_num = int(sliders[3].val)
        led_m = sliders[4].val
        pd_m = sliders[5].val
        background_log = sliders[6].val
        bandwidth_log = sliders[7].val
        led_alpha = np.deg2rad(sliders[8].val)
        pd_alpha = np.deg2rad(sliders[9].val)
        pd_saturate_log = sliders[10].val
        shunt_log = sliders[11].val
        gain = sliders[12].val

        led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
        background = 10**background_log
        bandwidth = 10**bandwidth_log
        pd_saturate = 10**pd_saturate_log
        shunt = 10**shunt_log
        
        solve_mulmul()
        count_total = np.nansum(error<tolerance)
        count_kpos = np.nansum(error<tolerance,axis=1)
        count_krot = np.nansum(error<tolerance,axis=0)
        effective_pos = count_kpos/krot >=effective/100
        effective_rot = count_krot/kpos >=effective/100
        print(np.nansum(count_kpos))
        sc1 = ax1.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap=colormap,norm = normalizep,alpha=0.5)
        sc2 = ax2.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
        sc3 = ax3.scatter(testp_pos[0,effective_pos],testp_pos[1,effective_pos],testp_pos[2,effective_pos],color = 'b',alpha=0.5)
        sc4 = ax4.scatter(testp_rot[2,effective_rot],np.rad2deg(testp_rot[1,effective_rot])  ,color = 'b')

        sliders[7].valtext.set_text(f'{bandwidth:.4E}')
        sliders[6].valtext.set_text(f'{background:.4E}')
        sliders[10].valtext.set_text(f'{pd_saturate:.4E}') 
        sliders[11].valtext.set_text(f'{shunt:.4E}')  

        fig.suptitle(f'餘容許範圍內的樣本點總數：{count_total}')
        
        global max_temp,max_led,max_pd,max_text
        if count_total>max_temp:
            print('----------------------')
            max_temp=count_total
            max_led = led_alpha
            max_pd = pd_alpha
            max_text.set_text(f'Max:{max_temp}')
            print('Nt:',max_temp)
            print('LED config:\n',np.rad2deg(led_alpha))
            print('PD config:\n',np.rad2deg(pd_alpha))
        #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
        # print(bandwidth)
        fig.canvas.draw_idle()
        plt.show()


    for i in np.arange(len(min_val)):
        #samp.on_changed(update_slider)
        sliders[i].on_changed(sliders_on_changed)
    plt.show()

elif mode == 'save':
    testp_pos,testp_rot = set_scenario(scenario)
    count = 0
    # ans = np.zeros((14,14,5,5,5,5,2))
    numl = np.array([3,5,8,10,15])
    nump = np.array([3,5,8,10,15])
    ml = np.array([1,1.5,2,3,5])
    mp = np.array([1,1.5,2,3,5])
    alphal = np.deg2rad(np.array([5,10,15,30,50]))
    alphap = np.deg2rad(np.array([5,10,15,30,50]))
    
    
    for led_num in numl:
        for pd_num in nump:
            for led_m in ml:
                for pd_m in mp:
                    for led_alpha in alphal:
                        for pd_alpha in alphap:
                            led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
                            
                            solve_mulmul()
                            error = error.filled(np.nan)
                            
                            np.save(f'./data_new/{led_num} {pd_num} {led_m} {pd_m} {np.round(np.rad2deg(led_alpha))} {np.round(np.rad2deg(pd_alpha))}.npy',error)
    
elif mode == 'analysis_graph':
        # initiate
    testp_pos ,testp_rot = set_scenario(scenario)
    kpos = testp_pos.shape[1]
    # testp_rot = np.array([[np.pi,0,0]]).T
    krot = testp_rot.shape[1]
    # effective = 80

    numl = np.array([3,5,8,10,15])
    nump = np.array([3,5,8,10,15])
    ml = np.array([1,1.5,2,3,5])
    mp = np.array([1,1.5,2,3,5])
    alphal = np.deg2rad(np.array([5,10,15,30,50]))
    alphap = np.deg2rad(np.array([5,10,15,30,50]))

    # for lambertian
    pd_num = 5
    led_num = 5
    
    pd_alpha = np.deg2rad(10)
    led_alpha = np.deg2rad(10)

    object1 = ml
    object2 = mp

    
    fig1 = plt.figure(figsize=(8, 8))
    fig2 = plt.figure(figsize=(8, 8))
    colormap= plt.cm.get_cmap('YlOrRd')
    normalizep =  colors.Normalize(vmin=0, vmax=krot)
    normalizer =  colors.Normalize(vmin=0, vmax=kpos)

    fig1.subplots_adjust(wspace=0.5,hspace=0.5,right = 0.85,left = 0.1,top = 0.9)
    fig1.suptitle ('平移樣本點')
    fig2.subplots_adjust(wspace=0.8,hspace=0.8,right = 0.85,left = 0.13,top = 0.9)
    fig2.suptitle('旋轉樣本點')
    # text_ax1 = fig1.add_axes([0,0,0.2,1], frameon=False)
    # text_ax2 = fig1.add_axes([0,0.9,1,0.1], frameon=False)
    for i in range(len(object2)):
        fig1.text(0.02,0.85-i*0.7/(len(object2)-1),f'Mp={object2[i]}',fontsize = 10)
        fig1.text(0.12+i*0.8/len(object2),0.92,f'Ml={object1[i]}',fontsize = 10)
        fig2.text(0.02,0.85-i*0.7/(len(object2)-1),f'Mp={object2[i]}',fontsize = 10)
        fig2.text(0.12+i*0.8/len(object2),0.94,f'Ml={object1[i]}',fontsize = 10)

    
    
    for A in range(len(object1)):
        for B in range(len(object2)):
            led_m = object1[A]
            pd_m = object2[B]
            led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
            solve_mulmul()

            count_kpos = np.nansum(error<tolerance,axis=1)
            count_krot = np.nansum(error<tolerance,axis=0)
            effective_pos = count_kpos/krot >=effective/100
            effective_rot = count_krot/kpos >=effective/100

            ax1 = fig1.add_subplot(len(object1),len(object2),1+A*len(object1)+B,projection='3d')
            ax1.set_box_aspect(aspect = (1,1,1))

            # ax1.set_xlabel('x')
            # ax1.set_ylabel('y')
            # ax1.set_zlabel('z')
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

            
            ax2 = fig2.add_subplot(len(object1),len(object2),1+A*len(object1)+B,projection='polar')
    
            sc2 = ax2.scatter((testp_rot[2,:]),np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
            ax2.set_ylim([0,rot_max])
            # ax2.text(1,1,'pitch(degree)',rotation = 15)
            # ax2.text(np.deg2rad(60),80,'yaw(degree)')
    
    cbar_ax = fig1.add_axes([0.92, 0.15, 0.02, 0.7])       
    colorbar = fig1.colorbar(sc1, cax=cbar_ax)
    colorbar.ax.set_ylabel('容許範圍內的樣本點數量')
             
    cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    colorbar = fig2.colorbar(sc2, cax=cbar_ax)
    plt.show()
    
elif mode == 'config_interactive':
    
    max_temp = 0
    max_led = []
    max_pd = []
    
    # initiate
    testp_pos ,testp_rot = set_scenario(scenario)
    kpos = testp_pos.shape[1]
    # testp_rot = np.array([[np.pi,0,0]]).T
    krot = testp_rot.shape[1]
    # effective = 80 


    led_num = 10
    pd_num = 10
    
    led_m = 1
    pd_m = 1
    
    # led_alpha = np.deg2rad(42.3)
    # pd_alpha = np.deg2rad(45.2)
    led_alpha = np.deg2rad(50)
    pd_alpha = np.deg2rad(50)
    
    led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
    
    
    
    
    solve_mulmul()
    count_total = np.nansum(error<tolerance)
    count_kpos = np.nansum(error<tolerance,axis=1)
    count_krot = np.nansum(error<tolerance,axis=0)
    effective_pos = count_kpos/krot >=effective/100
    effective_rot = count_krot/kpos >=effective/100
    max_temp=count_total
    max_led = led_ori_ang
    max_pd = pd_ori_ang
    
    fig = plt.figure(figsize=(15, 8))
    sup = fig.suptitle(count_total)
    colormap= plt.cm.get_cmap('YlOrRd')
    normalizep =  colors.Normalize(vmin=0, vmax=krot)
    normalizer =  colors.Normalize(vmin=0, vmax=kpos)
    fig.subplots_adjust(left=0.05,right=0.9,wspace=0.3,hspace=0.3)
    
    ax1 = fig.add_subplot(2,4,1,projection='3d')
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
    
    
    
    ax2 = fig.add_subplot(2,4,5,projection='polar')
    
    sc2 = ax2.scatter((testp_rot[2,:]),np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
    colorbar = fig.colorbar(sc2,shrink=0.3,pad=0.15)
    colorbar.ax.set_ylabel('容許範圍內的樣本點數量')
    ax2.set_title('旋轉樣本點')
    ax2.text(1,1,'pitch(degree)',rotation = 15)
    ax2.text(np.deg2rad(60),80,'yaw(degree)')
    ax2.set_ylim([0,rot_max])
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    
    ax3 = fig.add_subplot(2,4,2,projection='3d')
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
    
    
    
    
    ax4 = fig.add_subplot(2,4,6,projection='polar')
    
    sc4 = ax4.scatter((testp_rot[2,effective_rot]),np.rad2deg(testp_rot[1,effective_rot])  ,color = 'b')

    ax4.set_title('旋轉樣本有效範圍')
    ax4.text(0,0,'pitch(degree)',rotation = 15)
    ax4.text(np.deg2rad(60),80,'yaw(degree)')
    ax4.set_ylim([0,rot_max])
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(sc, cax=cbar_ax)
    
    a = np.linspace(0,2*np.pi,20)
    b = np.linspace(0,1,10)
    r = np.outer(b, np.cos(a))
    o = np.outer(b, np.sin(a))
    zeror = np.zeros(r.shape)
    
    
    ax5 = fig.add_subplot(2,4,3,projection='3d')
    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])
    ax5.zaxis.set_ticklabels([])
    ax5.set_axis_off()
    ax5.set_box_aspect(aspect = (1,1,1))
    ax5.title.set_text('LED次系統')
    ax5.set_xlim3d(-1,1)
    ax5.set_ylim3d(-1,1)
    ax5.set_zlim3d(-1,1)
    ax5.quiver([0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1],color = 'grey')
    ax5.text(1.1,0,0,'x',color = 'grey')
    ax5.text(0,1.1,0,'y',color = 'grey')
    ax5.text(0,0,1.1,'z',color = 'grey')
    # ax5.text(0.5,0.5,-0.1,'z=0平面',color = 'grey')
    ax5.plot_surface(r,o,zeror, color="grey",alpha=0.25)
    

    u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,21))
    x = 1*np.cos(u)*np.sin(v)
    y = 1*np.sin(u)*np.sin(v)
    z = 1*np.cos(v)
    ax5.plot_wireframe(x, y, z, color="w",alpha=0.15, edgecolor="#808080")
    zero = np.zeros((led_num,))
    sc5 = ax5.quiver(zero,zero,zero,led_ori_car[0,:],led_ori_car[1,:],led_ori_car[2,:],color = 'b',label='LED指向')
    ax5.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2))

    ax6 = fig.add_subplot(2,4,7,projection='3d')
    ax6.xaxis.set_ticklabels([])
    ax6.yaxis.set_ticklabels([])
    ax6.zaxis.set_ticklabels([])
    ax6.set_axis_off()
    ax6.set_box_aspect(aspect = (1,1,1))
    ax6.title.set_text('PD次系統')
    ax6.set_xlim3d(-1,1)
    ax6.set_ylim3d(-1,1)
    ax6.set_zlim3d(-1,1)
    ax6.quiver([0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1],color = 'grey')
    ax6.text(1.1,0,0,'x',color = 'grey')
    ax6.text(0,1.1,0,'y',color = 'grey')
    ax6.text(0,0,1.1,'z',color = 'grey')
    ax6.plot_surface(r,o,zeror, color="grey",alpha=0.25)

    ax6.plot_wireframe(x, y, z, color="w",alpha=0.15, edgecolor="#808080")
    zero = np.zeros((pd_num,))
    sc6 = ax6.quiver(zero,zero,zero,pd_ori_car[0,:],pd_ori_car[1,:],pd_ori_car[2,:],color='firebrick',label='PD指向')
    ax6.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2))
        #text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
    #print(vec,dis)
    
    max_text = fig.text(0.8,0.1,f'Max:{max_temp}')

    sliders = []
    text = [r'LED朗博次方$M\ell$',r'PD朗博次方$Mp$']
    init_val = [led_m,pd_m]
    min_val = [1,10]
    max_val = [1,10]
    print(init_val)
    # for i in range(2):
    axamp = plt.axes([0.75, 0.9-(0*0.03), 0.1, 0.02])
    s = Slider(axamp, r'LED朗博次方$M\ell$', 1, 10, valinit=led_m)
    sliders.append(s)
    axamp = plt.axes([0.75, 0.9-(1*0.03), 0.1, 0.02])
    s = Slider(axamp, r'PD朗博次方$Mp$', 1, 10, valinit=pd_m)
    sliders.append(s)

    for i in range((led_num)):
        axamp1 = plt.axes([0.7, 0.8-(i*0.03), 0.1, 0.02])
        axamp2 = plt.axes([0.86, 0.8-(i*0.03), 0.1, 0.02])
        s = Slider(axamp1, r'$^L\alpha_{{{:2d}}}$'.format(i+1), 0, 180, valinit =np.rad2deg(led_ori_ang[0,i]))
        sliders.append(s)
        s = Slider(axamp2, r'$^L\beta_{{{:2d}}}$'.format(i+1), 0, 360,valinit = np.rad2deg(led_ori_ang[1,i]))
        sliders.append(s)
        #text.append((r'$^L\alpha_{{{:2d}}}$'.format(i+1)),(r'$^L\beta_{{{:2d}}}$'.format(i+1)))
    for i in range((pd_num)):
        axamp1 = plt.axes([0.7, 0.8-((i+led_num+1)*0.03), 0.1, 0.02])
        axamp2 = plt.axes([0.86, 0.8-((i+led_num+1)*0.03), 0.1, 0.02])
        s = Slider(axamp1, r'$^P\alpha_{{{:2d}}}$'.format(i+1), 0, 180, valinit = np.rad2deg(pd_ori_ang[0,i]))
        sliders.append(s)
        s = Slider(axamp2, r'$^P\beta_{{{:2d}}}$'.format(i+1), 0, 360,valinit = np.rad2deg(pd_ori_ang[1,i]))
        sliders.append(s)


    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):

        global  sc1,sc2,sc3,sc4,sc5,sc6,ax1,ax2,ax3,ax4,ax5,ax6,fig
        ax1.collections.remove(sc1)
        ax2.collections.remove(sc2)
        ax3.collections.remove(sc3)
        ax4.collections.remove(sc4)
        ax5.collections.remove(sc5)
        ax6.collections.remove(sc6)
        
        global led_m,pd_m,led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car
        # global pd_num,led_num,pd_m,led_m,pd_alpha,led_alpha,error,ledu,pdu,bandwidth,background,led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car
        led_m = sliders[0].val
        pd_m = sliders[1].val
        # tolerance = sliders[0].val
        # effective = sliders[1].val

        led_ori_ang = np.zeros((2,led_num))
        pd_ori_ang = np.zeros((2,pd_num))
        for i in range(led_num):
            led_ori_ang[0,i] = np.deg2rad(sliders[2+2*i].val)
            led_ori_ang[1,i] = np.deg2rad(sliders[2+1+2*i].val)
        for i in range(pd_num):
            pd_ori_ang[0,i] = np.deg2rad(sliders[2 + 2*led_num + 2*i].val)
            pd_ori_ang[1,i] = np.deg2rad(sliders[2 + 2*led_num + 1 + 2*i].val)

        
        led_ori_car = ori_ang2cart(led_ori_ang)
        pd_ori_car = ori_ang2cart(pd_ori_ang)

        solve_mulmul()
        count_total = np.nansum(error<tolerance)
        count_kpos = np.nansum(error<tolerance,axis=1)
        count_krot = np.nansum(error<tolerance,axis=0)
        effective_pos = count_kpos/krot >=effective/100
        effective_rot = count_krot/kpos >=effective/100
        fig.suptitle(count_total)
        global max_temp,max_led,max_pd,max_text
        if count_total>max_temp:
            print('----------------------')
            max_temp=count_total
            max_led = led_ori_ang
            max_pd = pd_ori_ang
            max_text.set_text(f'Max:{max_temp}')
            print('Nt:',max_temp)
            print('LED config:\n',led_ori_ang)
            print('PD config:\n',pd_ori_ang)
        
        sc1 = ax1.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap=colormap,norm = normalizep,alpha=0.5)
        sc2 = ax2.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
        sc3 = ax3.scatter(testp_pos[0,effective_pos],testp_pos[1,effective_pos],testp_pos[2,effective_pos],color = 'b',alpha=0.5)
        sc4 = ax4.scatter(testp_rot[2,effective_rot],np.rad2deg(testp_rot[1,effective_rot])  ,color = 'b')
        sc5 = ax5.quiver(np.zeros((led_num,)),np.zeros((led_num,)),np.zeros((led_num,)),led_ori_car[0,:],led_ori_car[1,:],led_ori_car[2,:],color='b')
        sc6 = ax6.quiver(np.zeros((pd_num,)),np.zeros((pd_num,)),np.zeros((pd_num,)),pd_ori_car[0,:],pd_ori_car[1,:],pd_ori_car[2,:],color='firebrick')
        # sliders[7].valtext.set_text(f'{bandwidth:.4E}')
        # sliders[6].valtext.set_text(f'{background:.4E}')
        # sliders[10].valtext.set_text(f'{pd_saturate:.4E}')  
        #ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
        print(count_total)
        fig.canvas.draw_idle()
        # plt.show()


    for i in np.arange(2+2*led_num+2*pd_num):
        #samp.on_changed(update_slider)
        sliders[i].on_changed(sliders_on_changed)
    

    plt.show()

    

elif mode == 'effect_plot':

    testp_pos ,testp_rot = set_scenario(scenario)
    kpos = testp_pos.shape[1]
    # testp_rot = np.array([[np.pi,0,0]]).T
    krot = testp_rot.shape[1]

    count = 0
    mlist = [1,1.5,2,5,7]
    numlist = [3,5,8,10,15]
    alphalist = np.deg2rad(np.array([5,10,15,20,30,40,50,60]))
    fig = [[],[],[]]

    # for m in range(len(mlist)):

    m=0
    alpha = 1
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1,projection='3d')

    led_m = mlist[m]
    pd_m = mlist[m]
    # led_alpha = alphalist[alpha]
    # pd_alpha = alphalist[alpha]

    meshl,meshp = np.meshgrid( np.rad2deg(alphalist),np.rad2deg(alphalist))
    # meshl,meshp = np.meshgrid(mlist,mlist)
    print(meshl,'meshl')
    surface = np.zeros((len(numlist),len(alphalist),len(alphalist)))
    # surface =  np.zeros((len(numlist),len(mlist),len(mlist)))

    for num in range(len(numlist)):
        led_num = numlist[num]
        pd_num = numlist[num]

        for l in range(len(alphalist)):
        # for l in range(len(mlist)):
            # for p in range(len(mlist)):
            for p in range(len(alphalist)):

                led_alpha = alphalist[l]
                pd_alpha = alphalist[p]
                # led_m = mlist[l]
                # pd_m = mlist[p]

                led_ori_ang,led_ori_car,pd_ori_ang,pd_ori_car = set_config(config_num, led_alpha, pd_alpha)
                solve_mulmul()  

                count_sol = np.nansum(error<tolerance)
                # print(alphalist[l],alphalist[p],count_sol)
                
                surface[num,l,p] = count_sol
                count = count+1
                print(count)
        
        sur = ax.plot_surface((meshl), (meshp), surface[num,:,:], label =  r'$L=P={{{:2d}}}$'.format(numlist[num]),alpha=0.7)
        sur._facecolors2d = sur._facecolor3d
        sur._edgecolors2d = sur._edgecolor3d
        
    np.save(f'./surface m{mlist[m]}.npy',surface)
    # np.save(f'./surface alpha{alphalist[alpha]}.npy',surface)


    ax.legend()
    # fig.suptitle(r'朗博次方($Mp,M\ell$)對系統成效的影響（$^L\alpha = ^P\alpha = {{{:.2f}}}(deg)$）'.format(np.rad2deg(alphalist[alpha])))
    # ax.set_xlabel(r'$M\ell$')
    # ax.set_ylabel(r'$Mp$')
    fig.suptitle(r'硬體天頂角($^P \alpha$,$^L\alpha$)對系統成效的影響（$Mp=M\ell={{{:2d}}}$）'.format(mlist[m]))
    ax.set_xlabel(r'$^L\alpha$(deg)')
    ax.set_ylabel(r'$^P\alpha$(deg)')
    ax.set_zlabel('於容許範圍內的樣本點數量')
    
    
    plt.show()

            



