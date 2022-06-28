import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import fsolve,root
import math
from mpl_toolkits import mplot3d
from geneticalgorithm import geneticalgorithm as ga
sympy.init_printing()

from funcfile import *

from geneticalgorithm import geneticalgorithm as ga
# plt.rcParams['font.sans-serif'] = ['Source Han Sans TW', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

pd_num = 5
led_num = 5


led_config =np.stack((  np.array([0.21473079, 2.40400875, 1.34644435, 0.22016245,0.33976713]), np.array([2.64807832, 2.45939339, 2.94233641, 4.74591118, 1.77008613])))         
pd_config = np.stack((np.array([0.35709663, 0.45533932, 0.88859295, 0.27061169, 1.89314697]),np.array([2.70758594,4.48369105, 3.14887709, 3.92191011, 3.68513225]))) 
led_x = np.multiply(np.cos(led_config[1,:]),np.sin(led_config[0,:]))
led_y = np.multiply(np.sin(led_config[1,:]),np.sin(led_config[0,:]))
led_z = np.cos(led_config[0,:])
pd_x = np.multiply(np.cos(pd_config[1,:]),np.sin(pd_config[0,:]))
pd_y = np.multiply(np.sin(pd_config[1,:]),np.sin(pd_config[0,:]))
pd_z = np.cos(pd_config[0,:])
zero = np.zeros(led_z.size)

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(211,projection='3d')
ax.title.set_text('LED組態')
ax.quiver(zero,zero,zero,led_x,led_y,led_z)
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)


ax = fig.add_subplot(212,projection='3d')
ax.title.set_text('PD組態')
ax.quiver(zero,zero,zero,pd_x,pd_y,pd_z)
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

# =============================================================================
# lamb = [1,2,5,10,20]
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(221,projection = 'polar')
# ax.title.set_text('輻射強度與入射角的關係')
# ax.set_xlabel(r'$\omega$')
# ax.set_ylabel(r'$\frac{I(\omega)}{I(\omega=0)}$')
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
# ax = fig.add_subplot(222,projection = 'polar')
# ax.title.set_text('朗博次方影響入射模式')
# ax.set_xlabel(r'$\omega$')
# ax.set_ylabel(r'$I(\omega)$')
# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# ax.set_theta_offset(.5*np.pi)
# for i in lamb:
#     theta = np.deg2rad(np.linspace(-90,90,200))
#     # dis = np.linspace(3,5,20)
#     # thetaa, diss = np.meshgrid(theta,dis)
#     # leng = thetaa.shape
#     I = (i+1)/2/np.pi * np.power(np.cos((theta)),i)
#     ax.plot(theta,I)
# ax.legend(legend, loc=(.85,.65))
# 
# plt.show()
# =============================================================================




# psi = np.zeros(leng)
# # print(thetaa.shape,diss.shape,psi.shape)
# pd = np.ones(3)
# led = np.array([func.lamb_order(10),1])



# p1 = func.src2pdcurrent(thetaa,psi,diss,pd,led)


# fig = plt.figure()
# ax3d = plt.axes(projection="3d")

# ax3d = plt.axes(projection='3d')
# ax3d.plot_surface(thetaa,diss,p1, cmap='plasma')
# ax3d.set_title('Radiant Flux at different distance and angle')
# ax3d.set_xlabel('LED出射角')
# ax3d.set_ylabel('距離')
# ax3d.set_zlabel('Relative radiant flux(%)')
# plt.show()

'''
a = np.array([[1,2]])
c = np.repeat(a,5,axis=0)
# print(np.concatenate((a,c),axis=1))
b = (np.array([[1,2],[3,4],[5,6]]))
print(np.concatenate((c,b),axis=0))
alpha_bound = np.repeat(np.array([[0,np.pi]]),5,axis=0)
beta_bound = np.repeat(np.array([[0,2*np.pi]]),5,axis=0)

d = np.concatenate((np.array([[2,70],[2,70]]),\
                    np.repeat(np.array([[0,np.pi]]),led_num,axis=0),\
                    np.repeat(np.array([[0,2*np.pi]]),led_num,axis=0),\
                    np.repeat(np.array([[0,np.pi]]),pd_num,axis=0),\
                    np.repeat(np.array([[0,2*np.pi]]),pd_num,axis=0)\
                    ),axis=0)
print(d)'''
'''
def rodrigue(k,ang):
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    I = np.eye(3)
    R = I + np.sin(ang)*K + (1-np.cos(ang))*(np.matmul(K,K)) #3x3
    return R

o = [0,0]
# print(np.deg2rad(45)*np.ones((1,100)))
circle1 = np.row_stack((  np.deg2rad(45)*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
circle1_cart = func.ori_ang2cart(circle1)
rot1 = rodrigue(np.array([1,0,0]),np.deg2rad(30))
# print(circle_cart)
# print(circle)
circle1_rot = np.matmul(rot1,circle1_cart)
o1 = np.matmul(rot1,np.array([0,0,1]))

circle2 = np.row_stack((  np.deg2rad(20)*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
circle2_cart = func.ori_ang2cart(circle2)
rot2 = rodrigue(np.array([0,1,1]),np.deg2rad(38))
circle2_rot = np.matmul(rot2,circle2_cart)
o2 = np.matmul(rot2,np.array([0,0,1]))


# circle3 = np.row_stack((  np.deg2rad(20)*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
# circle3_cart = func.ori_ang2cart(circle3)
# o3 = np.array([0,0,1])
# k = []
# theta = np.deg2rad()






fig = plt.figure()
ax = fig.add_subplot( projection='3d')
ax.set_box_aspect(aspect = (1,1,0.5))
# ax.set_aspect("auto")

# draw sphere
u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi/2,20))
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

a,b,c = circle1_rot
ax.plot(a,b,c,c='r')
ax.scatter(o1[0],o1[1],o1[2],marker='x',c='r')
a,b,c = circle2_rot
ax.plot(a,b,c,c='g')
ax.scatter(o2[0],o2[1],o2[2],marker='x',c='g')
# a,b,c = circle3_cart
# ax.plot(a,b,c,c='b')
# ax.scatter(o3[0],o3[1],o3[2],marker='x',c='b')


# ax3d.set_title('Radiant Flux at different distance and angle')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x, y, z = np.array([0,0,0])
u, v, w = np.array([0,0,1.5])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
ax.grid(False)
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_zlim(0,1.5)


plt.show()

# power = 100
# theta = np.linspace(-45,45,200)
# dis = np.linspace(3,5,20)
# thetaa, diss = np.meshgrid(theta,dis)
# leng = thetaa.shape

# psi = np.zeros(leng)
# # print(thetaa.shape,diss.shape,psi.shape)
# pd = np.ones(3)
# led = np.array([func.lamb_order(10),1])

'''


'''
x = np.linspace(0,80,100)
# y = np.linspace(0,45,100)
# xx,yy = np.meshgrid(cos1,cos2)
# ratio = np.linspace(0.1,10,100)



ratio = 0.5
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)


ratio = 1.1
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)

ratio = 1.5
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)

ratio = 2
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)


ratio = 3
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)

plt.xlabel("x(degree)")
plt.ylabel("y(degree)")
plt.title('cosx/cosy對應角度')
plt.legend(['0.5','1.1','1.5','2','3'])
# plt.colorbar()

# fig = plt.figure()
# ax3d = plt.axes(projection="3d")

# ax3d = plt.axes(projection='3d')
# # ax3d.plot_surface(xx,yy,p1, cmap='plasma')
# plt.contourf(xx,yy,p1)
# ax3d.set_title('Radiant Flux at different distance and angle')
# ax3d.set_xlabel('x')
# ax3d.set_ylabel('y')
# ax3d.set_zlabel('Relative radiant flux(%)')
plt.show()
'''