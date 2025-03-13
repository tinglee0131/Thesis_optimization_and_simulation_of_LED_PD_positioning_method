


import numpy as np
import matplotlib.pyplot as plt
import funcfile_old as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

# 假設靜態，因此單點對環境

# # pd coor x1 不會動
# led 用scenario產生很多點
# test point個transfer matrix(homo)

# config for pd &led
# transfer led coor to pd coor
# estimate d,theta,psi
# calculate strendth (mxn)

# calculate pos
# calculate error
# objective

# update

from funcfile import *


# =======================================================================
# Scenario

xx,yy,zz = np.array(np.meshgrid(np.arange(-1.5, 1.6, 1), #2 x
                      np.arange(-1.5, 1.6, 1), #1 y
                      np.arange(1, 3, 1))) #3 z

# [[x][y][z]] 3x?
testp_pos = np.stack((np.ndarray.flatten(xx),np.ndarray.flatten(yy),np.ndarray.flatten(zz)),0)                     
kpos = testp_pos[0].size # num of test position
print(kpos,'kpos')
# [[rotx][roty][rotz]] 3x?
# testp_rot = np.stack((np.deg2rad(np.arange(180-30,180+30,15)),np.zeros(np.arange(180,270,15).size),np.zeros(np.arange(180,270,15).size)),0)
testp_rot = np.array([[np.pi,0,0]]).T
krot = testp_rot[0].size # num of test rotate orientation
print(krot,'krot')

snr_db = 10
bias_val = 0

threshold_strength = 0 #pd訊號大於他才要被計算



# -------------------- inside optimization loop -------------------------
# =======================================================================
# pd led: coor config
# wrt 自己的coordinate
pd_num = 5
pd_pos = (np.zeros((3,pd_num)))
# pd_pos = np.tile(np.array([1,0,0]),(pd_num,1)).T
alpha = np.deg2rad(45)#傾角
beta = np.deg2rad(360/pd_num)#方位角
pd_ori_ang = np.stack( (alpha*np.ones(5),(beta*np.arange(1,pd_num+1))),0 )
pd_ori_car = ori_ang2cart(pd_ori_ang)

#pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]
pd_para = [2,1,1] #[0:M, 1:area, 2:respons]

led_num = 1
led_pos = form([[0,0,0]])
led_ori_ang = np.array([np.deg2rad([0,0])]).T
led_ori_car = ori_ang2cart(led_ori_ang)
led_para = [2,100]#led_para[0:m, 1:optical power]
# print(led_ori_car,'a')
# =======================================================================
# transfer led coor to pd coor

#(krot,kpos,3,m) 
#先把led_num個pos位置經過krot次旋轉，變成krotx3xled_num的testpoints，再將所有testpoints平移到testp_pos上
glob_led_pos = global_testp_trans(global_testp_after_rot(led_pos,testp_rot), testp_pos)
# print(glob_led_pos.shape)
#(krot,3,led_num) 
glob_led_ori = global_testp_after_rot(led_ori_car,testp_rot)

# =======================================================================
# estimate d,theta,psi

# krot x kpos x led x pd
dis,in_ang,out_ang = cal_d_in_out(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car,krot,kpos,led_num,pd_num)


'''check if d,in_ang,out_ang is right'''
'''
#krot2 kpos98 led2 pd5
a,b,c,d = 2,50,1,2

dis_cal,in_cal,out_cal = dis[a,b,c,d],in_ang[a,b,c,d],out_ang[a,b,c,d]

dis_real = np.sqrt(np.sum(np.square(glob_led_pos[a,b,:,c]-pd_pos[:,d])))
in_real = np.arccos(np.dot(glob_led_pos[a,b,:,c]-pd_pos[:,d],pd_ori_car[:,d])/dis_real)
out_real = np.arccos( np.dot( -glob_led_pos[a,b,:,c]+pd_pos[:,d],glob_led_ori[a,:,c] ) /dis_real )
print(dis_cal==dis_real)
print(in_cal==in_real)
print(out_cal==out_real)
# print(np.rad2deg(in_cal),np.rad2deg(in_real))
# print(np.rad2deg(out_cal),np.rad2deg(out_real))
'''




# =======================================================================
# calculate strendth 

# # dis,in_ang,out_ang   [krot x kpos x led x pd]
# strength 是pd電流 [krot x kpos x led x pd] 
strength = cal_strength_current(dis,in_ang,out_ang,led_para,pd_para) 

# =======================================================================
# add noise

# with db noise
# 10log_10(signal/noise)
# strength_wnoise = strength + (strength/(np.power(10,snr_db/10))) + bias(strength,bias_val)
strength_wnoise = strength

# =======================================================================
'''
# clear data

# view angle
# sarutation
# 忽略太小的數據
'''
# =======================================================================
# calculate pos
# 忽略太小的數據

'''假設pd於同個位置
Ax = b solve for best x
x_opt = (AtA)^(-1)Atb
'''

# def sol_pos_assume_pdp(strength_wnoise, led_para,pd_para):
    # strength_wnoise [krot x kpos x led x pd] 
    # pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]

pd_m, area, respon = pd_para
led_m , power= led_para
k = respon*power*(led_m+1)*area /(2*np.pi)
r,p = 0,51
# print(strength[r,p,:],'str')
# print(glob_led_pos[r,p,:,:],'pos')
# print(glob_led_ori[r,:,:],'ori')
# print(dis[r,p,:,:],'dis')
# print(in_ang[r,p,:,:],'inang')
# print(out_ang[r,p,:,:],'outang')

'''
def positive(lst):
    return [i for i in range(len(lst)) if lst[i] > 0] or None

set_of_signal = strength_wnoise[r,p,:,:] #ledxpd
for l in range(led_num): 
    # Ax = b
    # 處理掉太小的資訊
    # print(set_of_signal)
    big_enough_index= positive(set_of_signal[l,:]-threshold_strength)
    big_enough_len = len(big_enough_index)

    if big_enough_len>=4:
        # 可以得解
        # 夠大的強度
        # big_enough_list = set_of_signal[l,big_enough_index]

        # 取reference pd
        ref_index = big_enough_index[np.argmax(np.dot(pd_ori_car[:,big_enough_index].T,np.array([[0,0,1]]).T))] #最垂直的pd當作被除數
        # print(big_enough_index,'index')
        # print(big_enough_index[np.argmax(np.dot(pd_ori_car[:,big_enough_index].T,np.array([[0,0,1]]).T))],'dot')
        big_enough_index = np.delete(big_enough_index,ref_index)# bigenough-1個，少較ref
        print(big_enough_index,'new',ref_index)
        # A = np.zeros((big_enough_len-1,3))
        # b =  np.zeros((big_enough_len-1,1))
        Q = np.power(np.divide(set_of_signal[l,big_enough_index],set_of_signal[l,ref_index]) ,1/pd_m)# bigenough-1
        # Q = np.divide(set_of_signal[l,big_enough_index],set_of_signal[l,ref_index]) # bigenough-1
        print(Q,'Q')
        print(np.divide(np.cos(in_ang[r,p,l,big_enough_index]),np.cos(in_ang[r,p,l,ref_index])))
        print(big_enough_len)
        print(pd_ori_car[ref_index])
        A = pd_ori_car[:,big_enough_index].T-np.multiply(np.tile(Q,(3,1)).T,np.tile(pd_ori_car[:,ref_index],(big_enough_len-1,1)))
        b = np.multiply((-np.tile(led_pos[:,l],(big_enough_len-1,1))\
                            +pd_pos[:,big_enough_index].T),\
                            pd_ori_car[:,big_enough_index].T)\
                        .sum(axis = 1).reshape((big_enough_len-1,1))\
            + np.multiply(Q.reshape((big_enough_len-1,1)),\
                            np.dot(led_pos[:,l],pd_ori_car[:,ref_index])
                            -np.dot(pd_pos[:,ref_index],pd_ori_car[:,ref_index]))
        sol_pos = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
        print(A,'A')
        print(b,'b')
        print(np.multiply((-np.tile(led_pos[:,l],(big_enough_len-1,1))\
                            +pd_pos[:,big_enough_index].T),\
                            pd_ori_car[:,big_enough_index].T)\
                        .sum(axis = 1).reshape((big_enough_len-1,1)))
        print(sol_pos,'sol_pos',r,p,l)    


    else: print(r,p,l,'不夠多啦')



# print(A.shape,'A')
# print(b.shape,'b')
        


print(testp_pos[:,p],'real')
'''

# =======================================================================
# calculate error

# 共有krotxkpos個testpoint，產生krotxkpos個estimated pos
# estimated_pos [krot,kpos,3]
# 正確位置在testp_pos，因為glob_led_pos是led個別的位置不是整個coordinate的位置
# testp_pos [3xkpos], kpos一樣的話krot可忽視

estimated_pos = np.zeros((krot,kpos,3))
# testp_pos [3xkpos], kpos一樣的話krot可忽視
real_pos = np.tile(testp_pos.transpose(),(krot,1,1)) #[krotxkposx3]

# error measure only distance
error_dis = np.sqrt((np.square(estimated_pos-real_pos)).sum(axis=2))


# =======================================================================
# visualization
ax = fig = plt.figure().add_subplot(projection='3d')

# px,py,pz為三個list，分別紀錄所有要plot的位置分量
# pu, pv, pw為三個list，分別紀錄所有要plot的orientation分量

'''
px, py, pz = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))
pu = np.sin(np.pi * px) * np.cos(np.pi * py) * np.cos(np.pi * pz)
pv = -np.cos(np.pi * px) * np.sin(np.pi * py) * np.cos(np.pi * pz)
pw = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * px) * np.cos(np.pi * py) *
     np.sin(np.pi * pz))
'''
# Make the grid
px, py, pz = pd_pos
# print(px.shape)
# Make the direction data for the arrows
pu,pv,pw = ori_ang2cart(pd_ori_ang)

ax.quiver(px, py, pz, pu, pv, pw, length=0.2, normalize=True,color='g')

# Make the grid
px2, py2, pz2 = glob_led_pos.transpose(2,0,1,3)
# print(px2.shape)
# Make the direction data for the arrows
pu2,pv2,pw2 = np.tile(glob_led_ori,(kpos,1,1,1)).transpose(2,1,0,3)
# pu2,pv2,pw2 =np.ones((3,1,98,1))
ax.quiver(px2, py2, pz2, pu2, pv2, pw2, length=0.2, normalize=True,color='r')

# setting
# ax.set_xlim([0, 3])
# ax.set_ylim([0, 3])
# ax.set_zlim([0, 3])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("Test Points")
plt.grid()
plt.show()

# 分開比較：
# n,m數量

# 每次update:
# n,m pos,ori

print('hi')





