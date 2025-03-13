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

fig = plt.figure(figsize=(4, 3))


l1 = [1.344,	1.5	,1.5	,1.187,	1.292,	1.865]
p1 = [1.656	,2.072,	2.333,	2.177	,2.177,	2.23]
l3 = [1.187	,1.24	,1.24,	1.24	,1.5	,1.5]
p3 = [1.812,1.396,	1.5,	1.812	,2.33	,2.385]


led_list=np.arange(3,9,1)
ax = fig.add_subplot(1,1,1)
# for i in range()scene_c,label='最佳解的系統成效Nt')
ax.plot(led_list,l1,label='To=0.01的LED朗博次方',color='b')
ax.plot(led_list,l3,label='To=0.03的LED朗博次方',color='b',linestyle='dashed')
ax.plot(led_list,p1,label='To=0.01的PD朗博次方',color='r')
ax.plot(led_list,p3,label='To=0.03的PD朗博次方',color='r',linestyle='dashed')
# ax.plot(led_list,10000*np.ones(led_list.shape),label='To=0.05',color='g')
ax.set_ylabel('最佳朗博次方M')
ax.set_xlabel('LED與PD的硬體總數L=P')
ax.grid()
ax.legend(loc = 'upper left')



scene_a = [\
[39522,42216,42828,43765,43777,43265],\
[43792,45279,47893,48378,48844,48377],\
[47483,48005,52609,53191,53465,54144],\
[50511,50677,54010,56550,56272,56685]]
scene_a=np.array(scene_a)

scene_c = [2149,2544,3439,4512,8571,13677,14151,14453,14534,14614,14624]
sceneb1 = [513	,	952		,1195	,	1620	,	1714	,	2169	]
sceneb2 = [5123		,5747	,	5878		,5893	,	5915	,	6172]
#513		952		1195		1620		1714		2169	
# 5123		5747		5878		5893		5915		6172




# led_list=np.arange(3,9,1)
# ax = fig.add_subplot(1,1,1)
# # for i in range():
# #     ax.plot(np.arange(3,9,1),scene_a[i,:],label=f'LED總數 L={led_list[i]}')
# # ax.plot(led_list,scene_c,label='最佳解的系統成效Nt')
# ax.plot(led_list,sceneb1,label='To=0.01',color='r')
# ax.plot(led_list,sceneb2,label='To=0.03',color='b')
# ax.plot(led_list,10000*np.ones(led_list.shape),label='To=0.05',color='g')
# ax.set_ylabel('於容許範圍內的樣本點數量')
# ax.set_xlabel('LED與PD的硬體總數L=P')
# ax.grid()
# ax.legend(loc = 'upper left')




# led_list=np.arange(3,14,1)
# ax = fig.add_subplot(1,1,1)
# # for i in range():
# #     ax.plot(np.arange(3,9,1),scene_a[i,:],label=f'LED總數 L={led_list[i]}')
# ax.plot(led_list,scene_c,label='最佳解的系統成效Nt')
# ax.plot(led_list,15376*np.ones(led_list.shape),linestyle='dashed',color='grey',label='樣本總數K')
# ax.set_ylabel('於容許範圍內的樣本點數量')
# ax.set_xlabel('LED與PD的硬體總數L=P')
# ax.grid()
# ax.legend()


# led_list=[3,5,8,12]
# ax = fig.add_subplot(1,1,1)
# for i in range(4):
#     ax.plot(np.arange(3,9,1),scene_a[i,:],label=f'LED總數 L={led_list[i]}')

# ax.plot(np.arange(3,9,1),61000*np.ones(scene_a[0,:].shape),linestyle='dashed',color='grey',label='樣本總數K')
# ax.set_ylabel('於容許範圍內的樣本點數量')
# ax.set_xlabel('PD總數 P')
# ax.grid()
# ax.legend()


# color = colors.Normalize(vmin=0,vmax=61000)
# ax = fig.add_subplot(1,1,1,projection='3d')
# mesh1,mesh2=np.meshgrid(led_list,np.arange(3,9,1))
# print(mesh1.shape,scene_a.shape)
# sur = ax.plot_surface(mesh1,mesh2,scene_a.T,cmap = 'plasma',norm=color)
# # ax.plot(np.arange(3,9,1),61000*np.ones(scene_a[0,:].shape),linestyle='dashed',color='grey',label='樣本總數K')
# ax.set_zlabel('於容許範圍內的樣本點數量')
# ax.set_xlabel('LED總數 L')
# ax.set_ylabel('PD總數 P')
# ax.grid()
# fig.colorbar(sur,shrink=0.5,pad=0.15)
# # ax.legend()









# ---------------------

# ax = fig.add_subplot(1,1,1)

# to = [0.01, 0.05, 0.2,0.0230,\
#     0.0643,0.0843,0.0932,0.0733,0.0435\
#     ,0.388,0.5,0.3678,0.2879,0.2086]
# to_nt = [11776,41498,49383,27164,\
#     47479,48511,48797,48063,42695,50256\
#     ,50320, 50231 ,50140,49925 ]
# ax.scatter(to,to_nt)
# to = np.sort(np.array(to))
# to_nt = np.sort(np.array(to_nt))
# ax.set_xlabel('容許範圍(m)')

# to = [1000,370*10**3,10**6,10**9,1.6156*10**6\
#     ,1.487*10**7,1.078*10**8,1.08*10**5,9.88*10**3]
# to_nt = [49389,41496,36653,450,37792\
#     ,15729,3356,47504,49163]
# to = np.sort(np.array(to))[::-1]
# to_nt = np.sort(np.array(to_nt))
# ax.set_xlabel('頻寬(Hz)')
# ax.set_xscale('log')



# to = [2.471,2.714,2.297,\
#     2.037,1.782,1.493,\
#         1.14,1,1.395,\
#     1.32,1.262,1.198,1.065,1.094]
# to_nt = [442,434,846,\
#     1160,1951,5580,\
#         47079,49261,8811,\
#     13647,20707,32154,49139,48933]
# ax.scatter(to,to_nt)
# to = np.sort(np.array(to))[::-1]
# to_nt = np.sort(np.array(to_nt))
# ax.set_xlabel('多重路徑增益')


mlist = [1,1.5,2,5,7]
numlist = [3,5,8,10,15]
alphalist = np.deg2rad(np.array([5,10,15,20,30,40,50,60]))
mat = np.load('./surface m1.npy')
matm = np.load('./surface alpha0.34.npy')
# print(mat.shape)

# ## single m line
# to = (mlist)
# to_nt = np.zeros(5)
# for i in range(len(to)):
#     to_nt[i] = matm[1,i,i]
#     # ax.plot(to,to_nt,label=r'$^L \alpha={{{:2f}}}$'.format(alphalist[i]))
# # print(to_nt)
# ax.plot(to,to_nt)#,label=r'$^L \alpha={{{:2f}}}$'.format(alphalist[i]))
# # ax.scatter(to,to_nt)
# # ax.plot(np.rad2deg(alphalist),61000*np.ones(alphalist.shape),linestyle='dashed',color='grey',label='樣本總數K')
# ax.set_xlabel(r'$Mp=M\ell$')
# # ax.set_xlabel(r'$ ^P \alpha$(deg)')
# # ax.legend()

## multiline m variable
# J=[0,  1,2,3,4]
# for j in range(len(J)):
#     s = J[j]
#     to = (mlist)
#     to_nt = np.zeros((5,))
#     for i in range(len(to)):
#         to_nt[i] = mat[s,i,i]
        
#     ax.plot(to,to_nt,label=f'L=P={numlist[j]}')
#     # print(to_nt)
#     # ax.plot(np.rad2deg(alphalist),61000*np.ones(alphalist.shape),linestyle='dashed',color='grey',label='樣本總數K')
#     ax.set_xlabel(r'$Mp=M\ell$')
#     # ax.set_xlabel(r'$ ^P \alpha$(deg)')
#     ax.legend()

## single alpha line
# to = np.rad2deg(alphalist)
# to_nt = np.zeros(alphalist.shape)
# for i in range(len(to)):
#     to_nt[i] = mat[2,i,i]
#     # ax.plot(to,to_nt,label=r'$^L \alpha={{{:2f}}}$'.format(alphalist[i]))
# # print(to_nt)
# # ax.plot(np.rad2deg(alphalist),61000*np.ones(alphalist.shape),linestyle='dashed',color='grey',label='樣本總數K')
# ax.set_xlabel(r'$^L \alpha = ^P \alpha$(deg)')
# # ax.set_xlabel(r'$ ^P \alpha$(deg)')
# ax.legend()

## multiline alpha variable
# J=[0,  1,2,3,4]
# for j in range(len(J)):
#     s = J[j]
#     to = np.rad2deg(alphalist)
#     to_nt = np.zeros(alphalist.shape)
#     for i in range(len(to)):
#         to_nt[i] = mat[s,i,i]
        
#     ax.plot(to,to_nt,label=f'L=P={numlist[j]}')
#     # print(to_nt)
#     # ax.plot(np.rad2deg(alphalist),61000*np.ones(alphalist.shape),linestyle='dashed',color='grey',label='樣本總數K')
#     ax.set_xlabel(r'$^L \alpha = ^P \alpha$(deg)')
#     # ax.set_xlabel(r'$ ^P \alpha$(deg)')
#     ax.legend()


# ax.set_ylabel('於容許範圍內的樣本點數量')
# ax.plot(to,to_nt)
# ax.scatter(to,to_nt)



# =============================================
## draw surface
# J=[0,1,4]
# ax = fig.add_subplot(1,1,1,projection='3d')
# color = colors.Normalize(vmin=0,vmax=61000)
# for  j in range(3):
#     s = J[j]
#     # ax = fig.add_subplot(1,1,1,projection='3d')
#     ax.set_box_aspect(aspect = (1,1,0.5))
#     to = np.rad2deg(alphalist)
#     mesh1,mesh2 = np.meshgrid(to,to)
#     to_nt = np.zeros(mesh1.shape)
#     sur = ax.plot_surface(mesh1,mesh2,mat[s,:,:],label = f'L=P={numlist[s]}',alpha=0.8)#,cmap = 'plasma',norm=color)
#     sur._facecolors2d = sur._facecolor3d
#     sur._edgecolors2d = sur._edgecolor3d
#     ax.set_title(r'$L=P = {{{:2f}}}$'.format(numlist[s]))
    

#     # for i in range(len(to)):
#     #     to_nt[i] = mat[2,i,i]
#         # ax.plot(to,to_nt,label=r'$^L \alpha={{{:2f}}}$'.format(alphalist[i]))
#     # print(to_nt)
#     # ax.plot(np.rad2deg(alphalist),61000*np.ones(alphalist.shape),linestyle='dashed',color='grey',label='樣本總數K')
#     ax.set_xlabel(r'$^L \alpha$(deg)')
#     ax.set_ylabel(r'$^P \alpha$(deg)')
#     # ax.set_xlabel(r'$ ^P \alpha$(deg)')
#     ax.legend()
# # fig.colorbar(sur,shrink = 0.5,pad=0.15)

# color = colors.Normalize(vmin=0,vmax=61000)
# for  j in range(1):
#     s = 1
#     ax = fig.add_subplot(1,1,1+j,projection='3d')
#     ax.set_box_aspect(aspect = (1,1,0.5))
#     to = (mlist)
#     mesh1,mesh2 = np.meshgrid(to,to)
#     to_nt = np.zeros(mesh1.shape)
#     sur = ax.plot_surface(mesh1,mesh2,matm[s,:,:],cmap = 'plasma',norm=color)
#     ax.set_title(r'$L=P = {{{:2f}}}$'.format(numlist[s]))
#     fig.colorbar(sur,shrink = 0.5,pad=0.15)

#     # for i in range(len(to)):
#     #     to_nt[i] = mat[2,i,i]
#         # ax.plot(to,to_nt,label=r'$^L \alpha={{{:2f}}}$'.format(alphalist[i]))
#     # print(to_nt)
#     # ax.plot(np.rad2deg(alphalist),61000*np.ones(alphalist.shape),linestyle='dashed',color='grey',label='樣本總數K')
#     ax.set_xlabel(r'$M\ell$')
#     ax.set_ylabel(r'$Mp$')
#     # ax.set_xlabel(r'$ ^P \alpha$(deg)')
#     # ax.legend()



plt.show()