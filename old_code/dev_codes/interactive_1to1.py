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

# set environment
def solve_mulmul(testp_pos,testp_rot,led_num,pd_num,led_m,pd_m):
    threshold = 0.001
    led_num = int(led_num)
    pd_num = int(pd_num)
    led_m = int(led_m)
    pd_m = int(pd_m)
    #pd_num = 7
    #pd_m = 3
    pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    pd_alpha = np.deg2rad(35)#傾角
    pd_beta = np.deg2rad(360/pd_num)#方位角
    
    
    #led_num = 5
    # led_m = 10
    led_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    led_alpha = np.deg2rad(45)#傾角
    led_beta = np.deg2rad(360/led_num)#方位角
    
    pd_area = 0.1
    led_pt = 0.5
    pd_saturate = np.inf
    pd_respon = 0.5
    
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
    
    #testp_pos = np.array([[1,1,1]]).T # 3x?
    kpos = testp_pos.shape[1]
    #testp_rot = np.array([[np.pi,0,0]]).T
    krot = testp_rot.shape[1]
    
    #(kpos,krot,led_num,3)  # kpos krot m 3
    glob_led_pos = global_testp_trans(global_testp_after_rot(led_pos,testp_rot), testp_pos)
    glob_led_ori = np.tile(global_testp_after_rot(led_ori_car,testp_rot), (kpos,1,1,1)).transpose((0,1,3,2))
    
    glob_inv_pd_pos = testp_rot_matlist(-testp_rot)
    glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,(kpos,1,1,1))-np.tile(glob_inv_pd_pos@testp_pos,(pd_num,1,1,1)).transpose(3,1,2,0)).transpose(0,1,3,2)
    
    
    
    
    # krot,kpos,led_num,pd_num
    dis,in_ang,out_ang = interactive_btw_pdled(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car)
    
    
    # 在view angle外的寫nan
    
    in_ang_view = filter_view_angle(in_ang,pd_view)
    out_ang_view = filter_view_angle(out_ang,led_view)
    
    in_ang_view[np.cos(in_ang_view)<0]=np.nan
    out_ang_view[np.cos(out_ang_view)<0]=np.nan
    
    
    const = pd_respon * pd_area * led_pt * (led_num+1)/(2*np.pi)
    light = const * np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
    # light = np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
    light[np.isnan(light)] = 0
    
    
    # =============================================================================
    # 這裡處理加上noise的部分
    # =============================================================================
    boltz = 1.380649 * 10**(-23)
    temp_k = 300
    bandwidth = 300
    elec_charge = 1.60217663 * 10**(-19)
    shunt = 50
    background = 10**(-6)
    dark_current = 10**(-12)
    NEP = 10**(-16)
    
    thermal_noise = 4*temp_k*boltz*bandwidth/shunt
    
    noise = 1*np.sqrt(thermal_noise\
              + 2*elec_charge*bandwidth*(light+background+dark_current)\
                  ) #+ 2*elec_charge*bandwidth*dark
    # print(noise[0,0,0,0])
    light_noise = light + noise
    light_floor = NEP*np.floor_divide(light_noise, NEP)
    
    # -------以下是硬體部分------------------
    
    
    
    
    
    
    
    # filter掉訊號中小於threshold的部分：nan
    # krot,kpos,led_num,pd_num
    light_f = np.copy(light_floor)
    light_f[light_f <= threshold] = np.nan
    light_f[light_f >= pd_saturate] = np.nan
    
    
    # =============================================================================
    # 先處理單個sample point
    # =============================================================================
    
    light_f = light_f.squeeze() #led pd
    
    # print(np.nanmean(light_f))
    # print(light_f)
    # light_f = np.around(light_f,decimals =10)
    
    # =============================================================================
    # 判斷特定LED是否有>=三個PD接收（才能判斷方位）
    # =============================================================================
    
    
    
    led_usable = np.sum(~np.isnan(light_f),axis=1)>2 #led,
    pd_usable = np.sum(~np.isnan(light_f),axis =0 )>2#pd,
    #pd_usable[2]=False
    light_led = light_f[led_usable,:] #ledu, pd
    light_pd = light_f[:,pd_usable] #led, pdu
    # =============================================================================
    # 取強度最大者作為ref1_led，建立平面的基準
    # 並利用maskled將light_led分成ref和other
    # => 計算ratio_led
    # =============================================================================
    ledu = led_usable.sum()
    pdu = pd_usable.sum()
    print(ledu,pdu)
    
    if (ledu<1 or pdu<1):
# =============================================================================
#         print('888')
# =============================================================================
        return None,None,None,None,None
    
    # print('Led, Pd usable amount: ',ledu,pdu)
    ref1_led = np.nanargmax(light_led, axis = 1) #ledu,
    ref1_pd = np.nanargmax(light_pd, axis = 0) #pdu,
    maskled = np.full(light_led.shape, False)
    maskled[np.arange(ledu),ref1_led] = True #ledu, pd
    maskpd = np.full(light_pd.shape, False)
    maskpd[ref1_pd,np.arange(pdu)] = True #led, pdu
    
    led_data_ref = light_led[maskled].reshape(-1,1)#ledu 1
    led_data_other = light_led[~maskled].reshape(ledu,-1)# ledu other
    pd_data_ref = light_pd[maskpd].reshape(1,-1)#1 pdu
    pd_data_other = light_pd[~maskpd].reshape(led_num-1,-1) #other, pdu
    # ref/other
    ratio_led = np.power(np.divide(led_data_ref, led_data_other),1/pd_m) #led_u x other
    ratio_pd = np.power(np.divide(pd_data_ref, pd_data_other),1/led_m) #other, pdu
    # in_ang  krot,kpos,led_num,pd_num
    
    
    # =============================================================================
    # 計算平面normal vector[ledu other 3]
    # =============================================================================
    #ledu x other x 3
    nor_led = np.tile(pd_ori_car.T,(ledu,1,1))[np.tile(maskled,(3,1,1)).transpose(1,2,0)].reshape(ledu,1,3)\
        - np.multiply(\
                    np.tile(pd_ori_car.T,(ledu,1,1))[np.tile(~maskled,(3,1,1)).transpose(1,2,0)].reshape(ledu,-1,3)\
                    ,ratio_led.reshape(ledu,-1,1))
    #led-1 x pdu x 3
    nor_pd = np.tile(led_ori_car,(pdu,1,1)).transpose(2,0,1)[np.tile(maskpd,(3,1,1)).transpose(1,2,0)].reshape(1,pdu,3)\
        - np.multiply(\
                    np.tile(led_ori_car,(pdu,1,1)).transpose(2,0,1)[np.tile(~maskpd,(3,1,1)).transpose(1,2,0)].reshape(-1,pdu,3)\
                    ,ratio_pd.reshape(-1,pdu,1))
    
    # 驗算dot
# =============================================================================
#     check_dot_led = (np.inner(np.array(testp_pos.T),nor_led))
#     check_dot_led = np.sum(~(np.isclose(check_dot_led,np.zeros(led_data_other.shape))|np.isnan(check_dot_led)))
#     check_dot_pd = (np.inner(np.tile(glob_inv_pd_pos[0,0,0,:],(1,1,1)),nor_pd))[0,:,:]
#     check_dot_pd = np.sum(~(np.isclose(check_dot_pd,np.zeros(pd_data_other.shape))|np.isnan(check_dot_pd)))
# =============================================================================
    # =============================================================================
    # print('-------------------------------------')
    # print('False normal vector from pd view:' ,check_dot_led)
    # print('False normal vector from led view:' ,check_dot_pd)
    # print('-------------------------------------')
    # 
    # =============================================================================
    
    # =============================================================================
    # 取led_data_other強度最大者作為ref2_led，當cross的基準
    # 並利用maskled2將data other分兩半
    # => 計算cross
    # =============================================================================
    ref2_led = np.nanargmax(led_data_other, axis = 1)
    ref2_pd = np.nanargmax(pd_data_other, axis = 0) #pdu,
    maskled2 = np.full(led_data_other.shape, False)
    maskled2[np.arange(ledu),ref2_led] = True #ledu, pd-1
    maskpd2 = np.full(pd_data_other.shape, False)
    maskpd2[ref2_pd, np.arange(pdu),] = True #led-1, pdu
    
    # 將normal vector分兩半
    nor_led_ref = nor_led[maskled2].reshape(-1,1,3) #ledu,1,3
    nor_led_other = nor_led[~maskled2].reshape(ledu,-1,3) #ledu,other-1,3
    nor_pd_ref = nor_pd[maskpd2].reshape(1,-1,3) #1,pdu,3
    nor_pd_other = nor_pd[~maskpd2].reshape(-1,pdu,3) #led-2,pdu,3
    # print(nor_pd_other)
    # =============================================================================
    # # 計算各平面交軸：cross vector
    # =============================================================================
    cross_led = np.cross(np.tile(nor_led_ref,(1,pd_num-2,1)),nor_led_other)#ledu,other-1,3
    cross_led = np.divide(cross_led, np.tile(np.sqrt(np.sum(np.square(cross_led),axis=2)),(1,1,1)).transpose(1,2,0))#ledu,other-1,3
    # cross_led_pstv：ref1的led指向，判斷軸方向是否正確
    cross_led_pstv = np.tile(pd_ori_car.T,(ledu,1,1))  [np.tile(maskled,(3,1,1)).transpose(1,2,0)].reshape((-1,3)) #ledu 3
    cross_led_mask = np.sum(np.multiply(cross_led, np.tile(cross_led_pstv,(pd_num-2,1,1)).transpose(1,0,2)),axis=2)<0#ledu other-1
    cross_led = np.where(np.tile(cross_led_mask,(3,1,1)).transpose(1,2,0),-cross_led,cross_led)
    #cross_led [ledu other-1 3]
    cross_pd = np.cross(np.tile(nor_pd_ref,(led_num-2,1,1)),nor_pd_other) #led-2,pdu,3
    cross_pd = np.divide(cross_pd, np.tile(np.sqrt(np.sum(np.square(cross_pd),axis=2)),(1,1,1)).transpose(1,2,0))#led-2,pdu,3
    cross_pd_pstv = (np.tile(led_ori_car,(pdu,1,1)).transpose(2,0,1))
    cross_pd_pstv = cross_pd_pstv [np.tile(maskpd,(3,1,1)).transpose(1,2,0)]    .reshape((1,-1,3)) #1,pdu 3
    cross_pd_mask = np.sum(np.multiply(cross_pd, np.tile(cross_pd_pstv,(led_num-2,1,1))),axis=2)<0#led-2 pdu 3
    cross_pd = np.where(np.tile(cross_pd_mask,(3,1,1)).transpose(1,2,0),-cross_pd,cross_pd)#led-2 pdu 3
    #cross_pd [led-2 pdu 3]
    cross_led_av = np.nanmean(cross_led,axis=(0,1))
    # print(cross_led_av.shape)
    # 驗算cross
# =============================================================================
#     check_cross_led = (np.sum(np.multiply(cross_led,np.tile(testp_pos.T/np.sqrt(np.sum(np.square(testp_pos))),(ledu,pd_num-2,1))),axis=2))
#     check_cross_led = np.sum(~(np.isnan(check_cross_led) | np.isclose(check_cross_led,np.ones((ledu,pd_num-2)))))
#     check_cross_pd = (np.sum(np.multiply(cross_pd,np.tile(-glob_inv_pd_pos[:,:,0,:]/np.sqrt(np.sum(np.square(glob_inv_pd_pos[:,:,0,:]))),(led_num-2,pd_usable.sum(),1))),axis=2))#led-2 pdu 3
#     check_cross_pd = np.sum(~(np.isnan(check_cross_pd) | np.isclose(check_cross_pd,np.ones((led_num-2,pdu)))))
# =============================================================================
    # =============================================================================
    # print('------------------------------------')
    # print('False cross vector from pd view:' ,check_cross_led)
    # print('False cross vector from pd view:' ,check_cross_pd)
    # print('------------------------------------')
    # =============================================================================
    
    
    # 答案求平均（忽略nan）
    ori_sol_pd_coor = np.nanmean(cross_led,axis = (0,1)) #3,
    ori_sol_led_coor = np.nanmean(cross_pd,axis = (0,1)) #3,
# =============================================================================
#     print(ori_sol_pd_coor,ori_sol_led_coor)
# =============================================================================
    
    
    # 由答案算in_ang,out_ang - ori_sol 3,  - ori_pd 3,pd
    sol_in_ang = np.arccos(np.inner(pd_ori_car.T,ori_sol_pd_coor)) # pd,
    sol_out_ang = np.arccos(np.inner(led_ori_car.T,ori_sol_led_coor)) #led,
    
    sol_dis = np.sqrt(const * np.divide(np.multiply(\
                                  np.tile(np.power(np.cos(sol_in_ang),pd_m),(led_num,1)),np.tile(np.power(np.cos(sol_out_ang),led_m),(pd_num,1)).T\
                     ),light_f))
    sol_dis_av = np.nanmean(sol_dis,axis=(0,1))
    print('dis',sol_dis_av)
    
    error = np.sqrt(np.sum(np.square(sol_dis_av*cross_led_av-glob_led_pos[0,0,0,:])))
# =============================================================================
#     error = (np.sum(np.square(np.multiply(cross_led_av,sol_dis_av.reshape(kpos,-1,1))-glob_led_pos[:,:,0,:]),axis=2))
#     error = error.filled(np.inf)
#     error[np.isclose(error,np.zeros(error.shape))] = np.nan 
#     error = np.sqrt(error)
#     error[np.isnan(error)]= 0
# =============================================================================
    # kp kr
    # print(ledu)
    # print(pdu)
    print('error:',error)

    return cross_led_av, sol_dis_av, ledu, pdu,error



# initiate
testp_pos = np.array([[0,1,1]]).T # 3x?
#kpos = testp_pos.shape[1]
testp_rot = np.array([[np.pi,0,0]]).T
#krot = testp_rot.shape[1]
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

# draw sphere
u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,20))
x = 0.1*np.cos(u)*np.sin(v)
y = 0.1*np.sin(u)*np.sin(v)
z = 0.1*np.cos(v)
sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

arrow = 0.5*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
ax.quiver(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=[0.2,0.5], color=["r",'g','b'])
arrow_rot = rotate_mat(testp_rot) @ arrow
axis_item = ax.quiver(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=0.1, color=["r",'g','b'])


vec, dis,ledu,pdu,error = solve_mulmul(testp_pos,testp_rot,led_num,pd_num,led_m,pd_m)
#ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='r')
if type(vec)!=type(None):
    ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
    text_item = ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error:.4E}')
else: 
    ans = ax.scatter(0,0,0,marker='x',color='k',s=10000)
    text_item = ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error}')
#text_num = ax.text2D(-0.14,-0.12,f'Led usable num:{ledu}\nPD usable num:{pdu}')
#print(vec,dis)


# Draw the initial plot
# The 'line' variable is used for modifying the line later
# =============================================================================
# [line] = ax.scatter(0,sol_in_ang, linewidth=2, color='red')
# ax.set_xlim([0, 1])
# ax.set_ylim([-10, 10])
# =============================================================================

# Add two sliders for tweaking the parameters
text = ['x','y','z','roll','pitch','yaw','led amount','pd amount','led m','pd m']
init_val = np.append(np.concatenate((testp_pos,testp_rot)).flatten(),(led_num,pd_num,led_m,pd_m))
min_val = [-1.5,-1.5,0,0,0,0,3,3,2,2]
max_val = [1.5,1.5,3,2*np.pi,2*np.pi,2*np.pi,20,20,70,70]
sliders = []
for i in np.arange(len(min_val)):

    axamp = plt.axes([0.74, 0.8-(i*0.05), 0.12, 0.02])
    # Slider
    # s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
    if i >5:
        s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i],valstep=1)
    else:
        s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
    sliders.append(s)


# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):

    global  sphere,axis_item,ans
    ax.collections.remove(sphere)
    ax.collections.remove(axis_item)
    ax.collections.remove(ans)
    #ax.collections.remove(text_num)
    arrow_rot = rotate_mat(np.array([sliders[3].val,sliders[4].val,sliders[5].val])) @ arrow
    sphere = ax.plot_wireframe(x+sliders[0].val, y+sliders[1].val, z+sliders[2].val, color="w",alpha=0.2, edgecolor="#808080")   
    axis_item = ax.quiver(sliders[0].val,sliders[1].val,sliders[2].val,arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=[0.2,0.5], color=["r",'g','b'])
    vec, dis,ledu,pdu,error = solve_mulmul(\
                    np.array([[sliders[0].val,sliders[1].val,sliders[2].val]]).T, np.array([[sliders[3].val,sliders[4].val,sliders[5].val]]).T,sliders[6].val,sliders[7].val,sliders[8].val,sliders[9].val)
    
    if type(vec)!=type(None):
        ans = ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
        text_item.set_text(f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error:.4E}')
    else: 
        ans = ax.scatter(0,0,0,marker='x',color='k',s=10000)
        text_item.set_text(f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error}')
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