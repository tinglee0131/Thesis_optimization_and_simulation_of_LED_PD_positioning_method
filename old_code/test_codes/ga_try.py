#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:19:37 2022

@author: tiffany
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geneticalgorithm import geneticalgorithm as ga

from funcfile import *
'''
def f(X):
    return np.sum(X)

varbound=np.array([[0.5,1.5],[1,100],[0,1]])
vartype=np.array([['real'],['int'],['int']])
model=ga(function=f,dimension=3,variable_type_mixed=vartype,variable_boundaries=varbound)

model.run()
'''

pd_num = 10
led_num = 10
threshold = 0.001
pd_area = 1
led_pt = 1
pd_saturate = np.inf
pd_respon = 1

tolerance = 0.05

# sample point
testp_pos = (np.mgrid[-1:1:4j, -1:1:4j, 1:3:4j].reshape(-1,4*4*4))#3x?
testp_rot = np.array([[0,0,0],[0,np.deg2rad(10),0],[0,np.deg2rad(-10),0],[np.deg2rad(10),0,0],[np.deg2rad(-10),0,0]]).T   +   np.array([[np.pi,0,0]]).T# 3x?
kpos = testp_pos.shape[1]
krot = testp_rot.shape[1]

bandwidth = 500
shunt = 50

def solve_mulmul(X):
    global pd_num
    global led_num # = 5
    led_m = X[0]
    pd_m = X[1]
    led_ori_ang = np.stack((X[2:2+led_num],X[2+led_num:2+2*led_num]))
    pd_ori_ang = np.stack((X[2+2*led_num:2+2*led_num+pd_num],X[2+2*led_num+pd_num:2+2*led_num+2*pd_num]))
    
    #,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,c1,c2,c3,c4,c5,d1,d2,d3,d4,d5): #int int list2x? list2x?
# set environment
    # led_conf = np.array([[a1,a2,a3,a4,a5],[b1,b2,b3,b4,b5]])
    # pd_conf = np.array([[c1,c2,c3,c4,c5],[d1,d2,d3,d4,d5]])
    global threshold # = 0.001
    
    # pd_num = int(pd_num)
    # led_num = int(led_num)
    # pd_m = int(pd_m)
    # led_m = int(led_m)
    
    # pd_m = 3
    pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    # pd_alpha = np.deg2rad(35)#傾角
    # pd_beta = np.deg2rad(360/pd_num)#方位角
    
    
    
    # led_m = 10
    led_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    # led_alpha = np.deg2rad(45)#傾角
    # led_beta = np.deg2rad(360/led_num)#方位角
    
    global pd_area #= 1
    global led_pt# = 1
    global pd_saturate# = np.inf
    global pd_respon #= 1
    # =============================================================================
    # ori_tar = np.deg2rad(np.array([[30,20]])).T #2x1
    # ori_tar_cart = ori_ang2cart(ori_tar)#3x1
    # tar_car_correct = ori_tar_cart
    # =============================================================================
    # print('0')
    # config
    pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
    # pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
    pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
    pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3
    
    led_pos = np.tile(np.array([[0,0,0]]).T,(1,led_num))
    # led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
    led_ori_car = ori_ang2cart(led_ori_ang) #3xled
    led_rot_mat = rotate_z_mul(led_ori_ang[1,:]) @ rotate_y_mul(led_ori_ang[0,:])#ledx3x3
    
    
    # sample point
    global testp_pos #= (np.mgrid[-1:1:4j, -1:1:4j, 1:3:4j].reshape(-1,4*4*4))#3x?
    # np.array([[0,1,1],[0,0,1],[0,-1,2]]).T # 3x?
    # kpos = testp_pos.shape[1]
    global testp_rot #= np.array([[np.pi,0,0],[np.pi,np.deg2rad(10),0]]).T
    # np.array([[np.pi,0,0],[0,np.pi,0]]).T
    # krot = testp_rot.shape[1]
    # testp_pos = np.array([[0,1,1],[0,0,1],[0,-1,2]]).T # 3x?
    global kpos #= testp_pos.shape[1]
    # testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
    global krot #= testp_rot.shape[1]
    
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
    
    # print('1')
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
    global bandwidth #= 300
    elec_charge = 1.60217663 * 10**(-19)
    global shunt #= 50
    
    noise = np.sqrt(4*temp_k*boltz*bandwidth/shunt\
              + 2*elec_charge*bandwidth*light\
                  ) #+ 2*elec_charge*bandwidth*dark
    # print(noise[0,0,0,0])
    light_noise = light + noise
                  
    
    # print('2')
    
    # -------以下是硬體部分------------------
    
    
    
    
    # filter掉訊號中小於threshold的部分：nan
    # krot,kpos,led_num,pd_num
    light_f = np.copy(light_noise)
    light_f[light_f <= threshold] = np.nan
    light_f[light_f >= pd_saturate] = pd_saturate
    
    
    # print('3')
    
    
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
    # print('4')
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
    # print('5')
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
    
    # print('6')
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
    
    # print('7')
    
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
    # print('8')
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
    # print('9')
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
    # print('10')
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
    
    global tolerance #= 0.05
    success = np.sum(error<tolerance)
    if success==0:
        error_av = 100
    else:
        error_av = np.mean(error[error<tolerance])
    
    # kp kr
    # print(ledu)
    # print(pdu)
    # print(error)
    # # print('12')
    # print(success)
    # print(error_av)
    return -success + error_av*10**8


algorithm_param = {'max_num_iteration': 500,\
                   'population_size':20,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


# print(solve_mulmul( [64.04126906 ,69.21085999]))

# def f(X):
#     return np.sum(X)
# print(solve_mulmul(1),'hi')
#(pd_m,led_m,pd_conf,led_conf): #int int list2x? list2x?
varbound = np.concatenate((np.array([[1,30],[1,30]]),\
                    np.repeat(np.array([[0,np.pi]]),led_num,axis=0),\
                    np.repeat(np.array([[0,2*np.pi]]),led_num,axis=0),\
                    np.repeat(np.array([[0,np.pi]]),pd_num,axis=0),\
                    np.repeat(np.array([[0,2*np.pi]]),pd_num,axis=0)\
                    ),axis=0)

# varbound = np.array([[1,30],[1,30]])
# varbound=np.array([[2,70],[2,70],[0,np.pi/4]])
# vartype=np.array([['real']])
model=ga(function=solve_mulmul,dimension=2+2*(led_num+pd_num),variable_type='real',variable_boundaries=varbound)
#2+2*(led_num+pd_num)
model.run()

# ans =  [1.11833894, 1.35157122, 0.21473079, 2.40400875, 1.34644435, 0.22016245,\
#  0.33976713, 2.64807832, 2.45939339, 2.94233641, 4.74591118, 1.77008613,\
#  0.35709663, 0.45533932, 0.88859295, 0.27061169, 1.89314697, 2.70758594,\
#  4.48369105, 3.14887709, 3.92191011, 3.68513225]
    
# print(solve_mulmul(ans))

                                                                           
# =============================================================================
# =============================================================================
# # testp_pos = (np.mgrid[-1:1:4j, -1:1:4j, 1:3:4j].reshape(-1,4*4*4))#3x?
# # testp_rot = np.array([[0,0,0],[0,np.deg2rad(10),0],[0,np.deg2rad(-10),0],[np.deg2rad(10),0,0],[np.deg2rad(-10),0,0]]).T   +   np.array([[np.pi,0,0]]).T# 3x?
# =============================================================================
#    The best solution found:
    # [1.59635534 3.64659549 0.33866975 0.00962708 0.10512115 2.36928225
#   0.01985999 3.09494779 0.21011412 2.79623198 0.30071889 2.41888534
#   0.55043657 2.4528098  4.59372575 0.33147949 1.73930968 0.22639113
#   3.21930293 0.81341371 5.2607883  1.85458207 0.05406089 0.57445071
#   1.83451829 0.03682501 2.55868568 0.17909927 1.70097813 0.13315306
#   2.024396   0.7498849  0.84769257 1.66959346 5.54648204 0.47771415
#   5.58602008 3.80653029 3.01512575 2.98516385 0.54550955 4.55192033]
# 
#   Objective function:
#   -317.74116345958845
# =============================================================================
