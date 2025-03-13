import numpy as np
import matplotlib.pyplot as plt
# import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
from funcfile import *
sympy.init_printing()
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

axis_color = 'lightgoldenrodyellow'



testp_pos = np.array([[0.5,1,2]]).T # 3x?
kpos = testp_pos.shape[1]
testp_rot = np.array([[np.pi,np.deg2rad(10),0]]).T
krot = testp_rot.shape[1]

#(kpos,krot,led_num,3)  # kpos krot m 3
pd_num = 3
pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
glob_inv_pd_pos = testp_rot_matlist(-testp_rot)
glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,(kpos,1,1,1))-np.tile(glob_inv_pd_pos@testp_pos,(pd_num,1,1,1)).transpose(3,1,2,0)).transpose(0,1,3,2)
glob_inv_pd_pos = glob_inv_pd_pos[0,0,0,:]





print(glob_inv_pd_pos)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(121,projection='3d')
ax.set_box_aspect(aspect = (1,1,0.5))
ax.grid(False)
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,1)


# Adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.25, bottom=0.25)






# sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="b",alpha=1, edgecolor="b")
# draw sphere
u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
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


alpha = np.deg2rad(np.array([30,30,50]))
beta = np.deg2rad(np.array([0,270,120]))
x = np.sin(alpha)*np.cos(beta)
y = np.sin(alpha)*np.sin(beta)
z = np.cos(alpha)
v = np.stack((x,y,z))
print(v.shape)
zero = np.array([0,0,0])

ax.quiver(zero[:3],zero[:3],zero[:3],x[:3],y[:3],z[:3],color = 'r')
ax.quiver(0,0,0,0,0,1.5,color='k',arrow_length_ratio=0.1)
ax.quiver(0,0,0,0,1.5,0,color='k',arrow_length_ratio=0.1)
ax.quiver(0,0,0,1.5,0,0,color='k',arrow_length_ratio=0.1)
# plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_axis_off()

d = np.sqrt(0.5**2+1+2**2)
ax.scatter(0.5/d,1/d,2/d,marker = 'x',s=100, color = 'g')

# inan1 = 
# circle =  np.stack((np.pi/2* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)# 3 x filt x sample
# a = cart2sph(n.reshape((3,1)))
# b = ori_ang2cart(a)
# print(np.rad2deg(a))
# print(b)
# m = rotate_mat((np.array([a[0,0],0,a[1,0]])))
# circle_cart = m@circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
inan1 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,0]))
inan2 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,1]))
inan3 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,2]))

du = testp_pos[:,0]/d
a,b,c = np.stack((du,-du)).T
ax.plot(a,b,c,color='g')
print(du,'du')
print(v[:,0],'du')
print(np.inner(testp_pos[:,0]/d,v[:,0]),np.rad2deg(inan1),np.rad2deg(inan2))

circle =  np.stack((inan1* np.ones((100)),\
    (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
circle_cart = ori_ang2cart(circle)
circle_cart = rotate_mat([0,alpha[0],beta[0]]) @circle_cart
ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)

circle =  np.stack((inan2* np.ones((100)),\
    (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
circle_cart = ori_ang2cart(circle)
circle_cart = rotate_mat([0,alpha[1],beta[1]]) @circle_cart
ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)

r12 = np.cos(inan2)/np.cos(inan1)
n = v[:,1]-r12*v[:,0]
print(n,'n')
# ax.quiver(0,0,0,n[0],n[1],n[2],color='b')
circle =  np.stack((np.pi/2* np.ones((100)),\
    (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
circle_cart = ori_ang2cart(circle)# 3 x filt x sample
n_ang = cart2sph(n.reshape((3,1)))
m = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]])))
circle_cart = m@circle_cart
ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
# circle_rot = pd_rot_mat[filt_l,:,:] @ circle_cart.transpose((1,0,2)) # filt x  3 x sample

a = np.linspace(-1,1,21)
b = np.linspace(-1,1,21)
A,B = np.meshgrid(a,b)
c = np.zeros((21,21))


m = np.stack((A.reshape((-1)),B.reshape((-1)),c.reshape((-1))))
a,b,c = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]]))) @ m
ax.plot_surface(a.reshape(21,21),b.reshape(21,21),c.reshape(21,21), color="b",alpha=0.2)

r13 = np.cos(inan3)/np.cos(inan1)
n = v[:,2]-r13*v[:,0]
n = -n
print(n,'n')
# ax.quiver(0,0,0,n[0],n[1],n[2],color='b')
circle =  np.stack((np.pi/2* np.ones((100)),\
    (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
circle_cart = ori_ang2cart(circle)# 3 x filt x sample
n_ang = cart2sph(n.reshape((3,1)))
m = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]])))
circle_cart = m@circle_cart
ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
# circle_rot = pd_rot_mat[filt_l,:,:] @ circle_cart.transpose((1,0,2)) # filt x  3 x sample

a = np.linspace(-1,1,21)
b = np.linspace(-1,1,21)
A,B = np.meshgrid(a,b)
c = np.zeros((21,21))


m = np.stack((A.reshape((-1)),B.reshape((-1)),c.reshape((-1))))
a,b,c = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]]))) @ m
ax.plot_surface(a.reshape(21,21),b.reshape(21,21),c.reshape(21,21), color="b",alpha=0.2)







ax = fig.add_subplot(122,projection='3d')
ax.set_box_aspect(aspect = (1,1,0.5))
ax.grid(False)
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,1)


# Adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.25, bottom=0.25)






# sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="b",alpha=1, edgecolor="b")
# draw sphere
u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
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


alpha = np.deg2rad(np.array([30,30,50]))
beta = np.deg2rad(np.array([0,270,120]))
x = np.sin(alpha)*np.cos(beta)
y = np.sin(alpha)*np.sin(beta)
z = np.cos(alpha)
v = np.stack((x,y,z))
print(v.shape)
zero = np.array([0,0,0])

ax.quiver(zero[0],zero[0],zero[0],x[0],y[0],z[0],color = 'r')
ax.quiver(zero[2],zero[2],zero[2],x[2],y[2],z[2],color = 'r')
ax.quiver(0,0,0,0,0,1.5,color='k')
ax.quiver(0,0,0,0,1.5,0,color='k')
ax.quiver(0,0,0,1.5,0,0,color='k')
# plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_axis_off()

d = np.sqrt(0.5**2+1+2**2)
ax.scatter(0.5/d,1/d,2/d,marker = 'x',s=100, color = 'g')

# inan1 = 
# circle =  np.stack((np.pi/2* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)# 3 x filt x sample
# a = cart2sph(n.reshape((3,1)))
# b = ori_ang2cart(a)
# print(np.rad2deg(a))
# print(b)
# m = rotate_mat((np.array([a[0,0],0,a[1,0]])))
# circle_cart = m@circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
inan1 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,0]))
inan2 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,1]))
inan3 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,2]))

du = testp_pos[:,0]/d

circle =  np.stack((inan1* np.ones((100)),\
    (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
circle_cart = ori_ang2cart(circle)
circle_cart = rotate_mat([0,alpha[0],beta[0]]) @circle_cart
ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)

circle =  np.stack((inan3* np.ones((100)),\
    (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
circle_cart = ori_ang2cart(circle)
circle_cart = rotate_mat([0,alpha[2],beta[2]]) @circle_cart
ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)

r13 = np.cos(inan3)/np.cos(inan1)
n = v[:,2]-r13*v[:,0]
n = -n
print(n,'n')
ax.quiver(0,0,0,n[0],n[1],n[2],color='b')
circle =  np.stack((np.pi/2* np.ones((100)),\
    (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
circle_cart = ori_ang2cart(circle)# 3 x filt x sample
n_ang = cart2sph(n.reshape((3,1)))
m = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]])))
circle_cart = m@circle_cart
ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
# circle_rot = pd_rot_mat[filt_l,:,:] @ circle_cart.transpose((1,0,2)) # filt x  3 x sample

a = np.linspace(-1,1,21)
b = np.linspace(-1,1,21)
A,B = np.meshgrid(a,b)
c = np.zeros((21,21))


m = np.stack((A.reshape((-1)),B.reshape((-1)),c.reshape((-1))))
a,b,c = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]]))) @ m
ax.plot_surface(a.reshape(21,21),b.reshape(21,21),c.reshape(21,21), color="b",alpha=0.2)


plt.show()
# =============================================================================
# axis_color = 'lightgoldenrodyellow'
# 
# 
# 
# testp_pos = np.array([[0.5,1,2]]).T # 3x?
# kpos = testp_pos.shape[1]
# testp_rot = np.array([[np.pi,np.deg2rad(10),0]]).T
# krot = testp_rot.shape[1]
# 
# #(kpos,krot,led_num,3)  # kpos krot m 3
# pd_num = 3
# pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
# glob_inv_pd_pos = testp_rot_matlist(-testp_rot)
# glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,(kpos,1,1,1))-np.tile(glob_inv_pd_pos@testp_pos,(pd_num,1,1,1)).transpose(3,1,2,0)).transpose(0,1,3,2)
# glob_inv_pd_pos = glob_inv_pd_pos[0,0,0,:]
# 
# 
# 
# 
# 
# print(glob_inv_pd_pos)
# 
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(121,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# 
# 
# # Adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.25, bottom=0.25)
# 
# 
# 
# 
# 
# 
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# # ax.plot_wireframe(x, y, z, color="b",alpha=1, edgecolor="b")
# # draw sphere
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# 
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)
# 
# 
# alpha = np.deg2rad(np.array([30,30,50]))
# beta = np.deg2rad(np.array([0,270,120]))
# x = np.sin(alpha)*np.cos(beta)
# y = np.sin(alpha)*np.sin(beta)
# z = np.cos(alpha)
# v = np.stack((x,y,z))
# print(v.shape)
# zero = np.array([0,0,0])
# 
# ax.quiver(zero[:2],zero[:2],zero[:2],x[:2],y[:2],z[:2],color = 'r')
# ax.quiver(0,0,0,0,0,1.5,color='k')
# ax.quiver(0,0,0,0,1.5,0,color='k')
# ax.quiver(0,0,0,1.5,0,0,color='k')
# # plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()
# 
# d = np.sqrt(0.5**2+1+2**2)
# ax.scatter(0.5/d,1/d,2/d,marker = 'x',s=100, color = 'g')
# 
# # inan1 = 
# # circle =  np.stack((np.pi/2* np.ones((100)),\
# #     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# # circle_cart = ori_ang2cart(circle)# 3 x filt x sample
# # a = cart2sph(n.reshape((3,1)))
# # b = ori_ang2cart(a)
# # print(np.rad2deg(a))
# # print(b)
# # m = rotate_mat((np.array([a[0,0],0,a[1,0]])))
# # circle_cart = m@circle_cart
# # ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
# inan1 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,0]))
# inan2 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,1]))
# inan3 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,2]))
# 
# du = testp_pos[:,0]/d
# print(du,'du')
# print(v[:,0],'du')
# print(np.inner(testp_pos[:,0]/d,v[:,0]),np.rad2deg(inan1),np.rad2deg(inan2))
# 
# circle =  np.stack((inan1* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)
# circle_cart = rotate_mat([0,alpha[0],beta[0]]) @circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)
# 
# circle =  np.stack((inan2* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)
# circle_cart = rotate_mat([0,alpha[1],beta[1]]) @circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)
# 
# r12 = np.cos(inan2)/np.cos(inan1)
# n = v[:,1]-r12*v[:,0]
# print(n,'n')
# ax.quiver(0,0,0,n[0],n[1],n[2],color='b')
# circle =  np.stack((np.pi/2* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)# 3 x filt x sample
# n_ang = cart2sph(n.reshape((3,1)))
# m = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]])))
# circle_cart = m@circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
# # circle_rot = pd_rot_mat[filt_l,:,:] @ circle_cart.transpose((1,0,2)) # filt x  3 x sample
# 
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# 
# 
# m = np.stack((A.reshape((-1)),B.reshape((-1)),c.reshape((-1))))
# a,b,c = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]]))) @ m
# ax.plot_surface(a.reshape(21,21),b.reshape(21,21),c.reshape(21,21), color="b",alpha=0.2)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ax = fig.add_subplot(122,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# 
# 
# # Adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.25, bottom=0.25)
# 
# 
# 
# 
# 
# 
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# # ax.plot_wireframe(x, y, z, color="b",alpha=1, edgecolor="b")
# # draw sphere
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# 
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)
# 
# 
# alpha = np.deg2rad(np.array([30,30,50]))
# beta = np.deg2rad(np.array([0,270,120]))
# x = np.sin(alpha)*np.cos(beta)
# y = np.sin(alpha)*np.sin(beta)
# z = np.cos(alpha)
# v = np.stack((x,y,z))
# print(v.shape)
# zero = np.array([0,0,0])
# 
# ax.quiver(zero[0],zero[0],zero[0],x[0],y[0],z[0],color = 'r')
# ax.quiver(zero[2],zero[2],zero[2],x[2],y[2],z[2],color = 'r')
# ax.quiver(0,0,0,0,0,1.5,color='k')
# ax.quiver(0,0,0,0,1.5,0,color='k')
# ax.quiver(0,0,0,1.5,0,0,color='k')
# # plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()
# 
# d = np.sqrt(0.5**2+1+2**2)
# ax.scatter(0.5/d,1/d,2/d,marker = 'x',s=100, color = 'g')
# 
# # inan1 = 
# # circle =  np.stack((np.pi/2* np.ones((100)),\
# #     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# # circle_cart = ori_ang2cart(circle)# 3 x filt x sample
# # a = cart2sph(n.reshape((3,1)))
# # b = ori_ang2cart(a)
# # print(np.rad2deg(a))
# # print(b)
# # m = rotate_mat((np.array([a[0,0],0,a[1,0]])))
# # circle_cart = m@circle_cart
# # ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
# inan1 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,0]))
# inan2 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,1]))
# inan3 = np.arccos(np.inner(testp_pos[:,0]/d,v[:,2]))
# 
# du = testp_pos[:,0]/d
# 
# circle =  np.stack((inan1* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)
# circle_cart = rotate_mat([0,alpha[0],beta[0]]) @circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)
# 
# circle =  np.stack((inan3* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)
# circle_cart = rotate_mat([0,alpha[2],beta[2]]) @circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'r',alpha=0.1)
# 
# r13 = np.cos(inan3)/np.cos(inan1)
# n = v[:,2]-r13*v[:,0]
# n = -n
# print(n,'n')
# ax.quiver(0,0,0,n[0],n[1],n[2],color='b')
# circle =  np.stack((np.pi/2* np.ones((100)),\
#     (np.linspace(0,2*np.pi,100))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle)# 3 x filt x sample
# n_ang = cart2sph(n.reshape((3,1)))
# m = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]])))
# circle_cart = m@circle_cart
# ax.plot(circle_cart[0,:],circle_cart[1,:],circle_cart[2,:],color = 'b')
# # circle_rot = pd_rot_mat[filt_l,:,:] @ circle_cart.transpose((1,0,2)) # filt x  3 x sample
# 
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# 
# 
# m = np.stack((A.reshape((-1)),B.reshape((-1)),c.reshape((-1))))
# a,b,c = rotate_mat((np.array([0,n_ang[0,0],n_ang[1,0]]))) @ m
# ax.plot_surface(a.reshape(21,21),b.reshape(21,21),c.reshape(21,21), color="b",alpha=0.2)
# =============================================================================



# =============================================================================
# axis_color = 'lightgoldenrodyellow'
# 
# 
# 
# testp_pos = np.array([[0.5,1,2]]).T # 3x?
# kpos = testp_pos.shape[1]
# testp_rot = np.array([[np.pi,np.deg2rad(10),0]]).T
# krot = testp_rot.shape[1]
# 
# #(kpos,krot,led_num,3)  # kpos krot m 3
# pd_num = 3
# pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
# glob_inv_pd_pos = testp_rot_matlist(-testp_rot)
# glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,(kpos,1,1,1))-np.tile(glob_inv_pd_pos@testp_pos,(pd_num,1,1,1)).transpose(3,1,2,0)).transpose(0,1,3,2)
# glob_inv_pd_pos = glob_inv_pd_pos[0,0,0,:]
# 
# 
# 
# 
# 
# print(glob_inv_pd_pos)
# 
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(221,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# 
# 
# # Adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.25, bottom=0.25)
# 
# 
# 
# 
# 
# 
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# # ax.plot_wireframe(x, y, z, color="b",alpha=1, edgecolor="b")
# # draw sphere
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# 
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)
# 
# 
# alpha = np.deg2rad(np.array([30,30,50]))
# beta = np.deg2rad(np.array([0,270,120]))
# x = np.sin(alpha)*np.cos(beta)
# y = np.sin(alpha)*np.sin(beta)
# z = np.cos(alpha)
# zero = np.array([0,0,0])
# 
# ax.quiver(zero,zero,zero,x,y,z,color = 'r')
# ax.quiver(0,0,0,0,0,1.5,color='k')
# ax.quiver(0,0,0,0,1.5,0,color='k')
# ax.quiver(0,0,0,1.5,0,0,color='k')
# # plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()
# 
# d = np.sqrt(0.5**2+1+2**2)
# ax.scatter(0.5/d,1/d,2/d,marker = 'x',s=100, color = 'g')
# 
# 
# 
# 
# # fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(222,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# 
# 
# x,y,z = 0 ,0,1
# a,b,c = np.pi,0,0
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(np.pi/6,np.pi/2,1))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# # ax.plot_wireframe(x, y, z, color="b",alpha=1, edgecolor="b")
# # draw sphere
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
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
# 
# arrow = 1.2*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
# 
# d = np.sqrt(np.sum(np.square(glob_inv_pd_pos)))
# x,y,z = glob_inv_pd_pos/d
# ax.scatter(x,y,z,marker = 'x',s=100,color = 'r')
# 
# 
# 
# alpha = np.deg2rad(np.array([30,30,45]))
# beta = np.deg2rad(np.array([0,200,100]))
# x = np.sin(alpha)*np.cos(beta)
# y = np.sin(alpha)*np.sin(beta)
# z = np.cos(alpha)
# zero = np.array([0,0,0])
# 
# ax.quiver(zero,zero,zero,x,y,z,color = 'g')
# ax.quiver(0,0,0,0,0,1.5,color='k')
# ax.quiver(0,0,0,0,1.5,0,color='k')
# ax.quiver(0,0,0,1.5,0,0,color='k')
# # plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()
# 
# 
# 
# 
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111,projection='3d')
# ax.set_box_aspect(aspect = (1,1,1))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.grid(True)
# ax.set_xlim3d(-1.5,1.5)
# ax.set_ylim3d(-1.5,1.5)
# ax.set_zlim3d(0,3)
# # ax.plot_wireframe(x+sliders[0].val, y+sliders[1].val, z+sliders[2].val, color="w",alpha=0.2, edgecolor="#808080")   
# 
# u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,20))
# x = 0.5*np.cos(u)*np.sin(v)
# y = 0.5*np.sin(u)*np.sin(v)
# z = 0.5*np.cos(v)
# ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# 
# a = np.linspace(-1,1,21)*0.5
# b = np.linspace(-1,1,21)*0.5
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)
# 
# arrow = 0.5*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
# ax.quiver(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=[0.2,0.5], color=["k"])
# a,b,c = rotate_mat(testp_rot) @ arrow
# ax.quiver(np.array([0.5,0.5,0.5]),np.array([1,1,1]),np.array([2,2,2]),a,b,c,arrow_length_ratio=[0.2,0.5], color=["k"])
# 
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# m = np.stack((A.reshape((-1,)),B.reshape((-1,)),c.reshape((-1,))))
# a,b,c = rotate_mat(testp_rot) @ m
# # ax.plot_surface(a.reshape((21,21)),b.reshape((21,21)),c.reshape((21,21)), color="grey",alpha=0.2)
# ax.quiver(0,0,0,0.5,1,2,color = 'b')
# # arrow_rot = rotate_mat(np.array([sliders[3].val,sliders[4].val,sliders[5].val])) @ arrow
# # sphere = ax.plot_wireframe(x+sliders[0].val, y+sliders[1].val, z+sliders[2].val, color="w",alpha=0.2, edgecolor="#808080")   
# # axis_item = ax.quiver(sliders[0].val,sliders[1].val,sliders[2].val,arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=[0.2,0.5], color=["r",'g','b'])
# # vec, dis,ledu,pdu,error = solve_mulmul(\
# #                 np.array([[sliders[0].val,sliders[1].val,sliders[2].val]]).T, np.array([[sliders[3].val,sliders[4].val,sliders[5].val]]).T,sliders[6].val,sliders[7].val,sliders[8].val,sliders[9].val)
# =============================================================================

# =============================================================================
# axis_color = 'lightgoldenrodyellow'
# 
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(121,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# 
# 
# # Adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.25, bottom=0.25)
# 
# t = np.arange(0.0, 1.0, 0.001)
# amp_0 = 5
# freq_0 = 3
# 
# x,y,z = 0 ,0,1
# a,b,c = np.pi,0,0
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(np.pi/6,np.pi/2,1))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="b",alpha=1, edgecolor="b")
# # draw sphere
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.5, edgecolor="#808080")
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)
# 
# arrow = 1.2*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
# 
# # ax.quiver(0,0,0,1,0,0,color='b')
# # ax.quiver(0,0,0,0,1,0,color='b')
# # ax.quiver(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=[0.2,0.5], color=["r",'g','b'])
# # arrow_rot = rotate_mat(testp_rot) @ arrow
# # axis_item = ax.quiver(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=0.1, color=["r",'g','b'])
# 
# I = np.cos(v)
# # sc = ax.scatter(x,y,z,I,c=I,cmap='rainbow',s=0.2)
# # fig.colorbar(surf)
# 
# ax.quiver(0,0,0,0,0,1.5,color='k')
# # plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()
# 
# 
# 
# ax = fig.add_subplot(122,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# 
# 
# # Adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.25, bottom=0.25)
# 
# t = np.arange(0.0, 1.0, 0.001)
# amp_0 = 5
# freq_0 = 3
# 
# x,y,z = 0 ,0,1
# a,b,c = np.pi,0,0
# 
# # draw sphere
# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi/2,20))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.5, edgecolor="#808080")
# 
# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)
# 
# arrow = 1.2*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
# 
# # ax.quiver(0,0,0,1,0,0,color='b')
# # ax.quiver(0,0,0,0,1,0,color='b')
# # ax.quiver(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=[0.2,0.5], color=["r",'g','b'])
# # arrow_rot = rotate_mat(testp_rot) @ arrow
# # axis_item = ax.quiver(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],arrow_rot[0,:],arrow_rot[1,:],arrow_rot[2,:],arrow_length_ratio=0.1, color=["r",'g','b'])
# 
# I = np.cos(v)
# sc = ax.scatter(x,y,z,I,c=I,cmap='rainbow',s=0.5)
# # fig.colorbar(surf)
# 
# ax.quiver(0,0,0,0,0,1.5,color='k')
# plt.colorbar(sc,shrink=0.3,location='right',pad=0.1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()
# =============================================================================




# =============================================================================
# # plt.xticks() 
# 
# lamb = [1,2,5,10,20]
# legend = ['M='+str(i) for i in lamb]
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(221,projection = 'polar')
# ax.title.set_text('朗博輻射模式\n輻射強度隨出入射角遞減')
# # ax.set_xlabel(r'$\omega$')
# # # ax.set_ylabel(r'$\frac{I(\omega)}{I(\omega=0)}$')
# # ax.set_ylabel(r'$\omega$', rotation=45, size=11)
# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# ax.set_theta_offset(.5*np.pi)
# for i in lamb:
#     theta = np.deg2rad(np.linspace(-90,90,200))
#     # dis = np.linspace(3,5,20)
#     # thetaa, diss = np.meshgrid(theta,dis)
#     # leng = thetaa.shape
#     I = np.power(np.cos((theta)),i)
#     ax.plot(theta,I)
# ax.legend(legend, loc=(.85,.65))
# 
# 
# legend = ['Ml='+str(i) for i in lamb]
# ax = fig.add_subplot(222,projection = 'polar')
# ax.title.set_text('LED朗博輻射模式\n相同總輻射能量下輻射強度與出射角的關係')
# # ax.set_xlabel(r'$\frac{I(\omega)}{I(\omega=0)}$')
# # ax.set_ylabel(r'$\omega$', rotation=45, size=11)
# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# 
# ax.set_theta_offset(.5*np.pi)
# temp = 0
# for i in lamb:
#     theta = np.deg2rad(np.linspace(-90,90,200))
#     # dis = np.linspace(3,5,20)
#     # thetaa, diss = np.meshgrid(theta,dis)
#     # leng = thetaa.shape
#     I = (i+1)/2/np.pi * np.power(np.cos((theta)),i)
#     ax.plot(theta,I)
#     temp = I
# ax.legend(legend, loc=(.85,.65))
# ax.set(yticks=np.arange(0,4,1))
# plt.show()
# =============================================================================

# =============================================================================
# power = 100
# theta = np.linspace(-45,45,200)
# dis = np.linspace(3,5,20)
# thetaa, diss = np.meshgrid(theta,dis)
# leng = thetaa.shape
# 
# psi = np.zeros(leng)
# # print(thetaa.shape,diss.shape,psi.shape)
# pd = np.ones(3)
# led = np.array([lamb_order(10),1])
# 
# 
# 
# p1 = src2pdcurrent(thetaa,psi,diss,pd,led)
# 
# 
# fig = plt.figure()
# ax3d = plt.axes(projection="3d")
# 
# ax3d = plt.axes(projection='3d')
# ax3d.plot_surface(thetaa,diss,p1, cmap='plasma')
# ax3d.set_title('Radiant Flux at different distance and angle')
# ax3d.set_xlabel('LED出射角')
# ax3d.set_ylabel('距離')
# ax3d.set_zlabel('Relative radiant flux(%)')
# plt.show()
# =============================================================================
