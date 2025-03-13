import numpy as np
import matplotlib.pyplot as plt
import funcfile_old as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()


class LED:
    def __init__(self,total_flux,lamb,light_freq,temp=25):
        self.total_flux = total_flux
        self.lamb = lamb
        self.light_freq = light_freq
        # self.flash_freq = flash_freq
        self.point = np.zeros(3)
        self.orien = np.array([0,0,-1])
        self.temp = temp
class PD:
    def __init__(self,resp,lamb,temp=25):
        self.resp = resp #responsivity
        self.lamb = lamb
        self.point = np.zeros(3)
        self.orien = np.zeros(3)
        self.temp = temp 

# src為一個光源，計算光源到多個感光點的距離
def disWsrc(src,pd_xyz):
    xx,yy,z = pd_xyz
    sz = xx.shape
    x_src = np.ones(sz)*src[0]
    y_src = np.ones(sz)*src[1]
    z_src = np.ones(sz)*src[2]
    return np.sqrt((xx-x_src)**2 + (yy-y_src)**2 + (z-z_src)**2)

# src為一個光源，計算光源到多個感光點的'receiver入射角'，回傳rad角度
def inangleWsrc(src,pd_xyzuvw): # nx ny nz是受光處的指向
    xx,yy,z ,nx,ny,nz = pd_xyzuvw
    xx = src[0]-xx #朝向src xyz座標的vector
    yy = src[1]-yy
    z = src[2]-z
    dot = np.multiply(xx,nx) + np.multiply(yy,ny) + np.multiply(z,nz) #dot product
    mag1 = np.sqrt(xx**2 + yy**2 + z**2)
    mag2 = np.sqrt(nx**2 + ny**2 + nz**2)
    cos = np.divide(dot, np.multiply(mag1,mag2))
    # print(np.rad2deg(np.arccos(cos)))
    return np.arccos(cos)

# src為一個光源，計算光源到多個感光點的'光源出射角'，回傳rad角度
def outangle(src_6,pd_xyz):
    src = src_6
    xx,yy,z = pd_xyz
    xx = xx - src[0] # 從src指向目標的vector
    yy = yy-src[1]
    z = z - src[2]
    dot = xx*src[3] + yy*src[4] + z*src[5]
    mag1 = np.sqrt(xx**2 + yy**2 + z**2) #vec
    mag2 = np.sqrt(src[3]**2 + src[4]**2 + src[5]**2 )#numeric
    cos = np.divide(dot,mag1*mag2)
    return  np.arccos(cos)

# 計算多個感光點的photocurrent，提供距離、出射入射角度
def src2current( theta, psi, dis, pd_para,led_para ): #pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]
    lambM, area, respon = pd_para
    m , power= led_para
    power_distri = ( m + 1 )/ (2*np.pi) * np.cos( (theta ) ) ** m
    received_pow = np.divide(np.multiply(power * power_distri * area, np.cos((psi))**lambM ),dis**2)
    pdcurrent =  respon*received_pow
    return pdcurrent

# tilted PD組態，給定傾角與數量，return指向
# ang(deg)
def generate_pd_tilt_config(num,ang):
    out = np.ones((num,3))
    out[:,2]=np.cos(np.deg2rad(ang))
    xy_dir = np.transpose(np.array(range(1,num+1))*2*np.pi/num)
    out[:,0] = np.sin(np.deg2rad(ang))*np.cos(xy_dir)
    out[:,1] = np.sin(np.deg2rad(ang))*np.sin(xy_dir)
    return out 


# 定義PD與LED object
# case: if pd together
pd_num = 4
pd_resp = 1
pd_lamb = 1
pd_point = np.array([0,0,-2])
pd_area = 1

led_num = 1
led_flux = 1000
led_lamb = 1
led_wavelength = 960*(10**(-14))
led_light_freq = 3*(10**8)/led_wavelength
led_sample_freq = 1000

#pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]
pd_para = np.array([pd_lamb,pd_area,pd_resp])
led_para = np.array([led_lamb,led_flux])

# 創建LED與PD
pd_orien_config = generate_pd_tilt_config(pd_num,30)
led_list = list()
led_save = np.zeros((pd_num,6))
for i in range(led_num):
    led_list.append(LED(led_flux,led_lamb,led_light_freq,led_sample_freq))
    led_list[0].point = np.array([0,0,2])
    led_save[i,:] = (np.concatenate((led_list[i].point,led_list[i].orien),axis=0))
pd_list = list()
pd_save = np.zeros((pd_num,6))
for i in range(pd_num):
    pd_list.append(PD(pd_resp,pd_lamb))
    pd_list[i].point = np.array([0,0,0])
    pd_list[i].orien = pd_orien_config[i]
    pd_save[i,:] = (np.concatenate((pd_list[i].point,pd_list[i].orien),axis=0))

dis = np.zeros((led_num,pd_num))
theta = np.zeros((led_num,pd_num))
phi = np.zeros((led_num,pd_num))
light = np.zeros((led_num,pd_num))


for i in range(led_num):
    dis[i,:] = disWsrc(led_save[i,:3], np.transpose( pd_save[:,:3]))
    theta[i,:] = outangle(led_save[i,:],np.transpose(pd_save[:,:3]))#led out(angle
    # print(pd_save[:,:])
    phi[i,:] = inangleWsrc(led_save[i,:3],np.transpose(pd_save[:,:]))
    light[i,:] = src2current(theta,phi,dis,pd_para,led_para)
# print(np.rad2deg(phi))
# print(light)




def equations(p):
    x, y, z = p
    F = np.empty(4)
    F[0] = ((x*pd_save[0][3]+y*pd_save[0][4]+z*pd_save[0][5])/(x*pd_save[1][3]+y*pd_save[1][4]+z*pd_save[1][5]))** pd_lamb - light[0][0]/light[0][1]
    F[1] = ((x*pd_save[1][3]+y*pd_save[1][4]+z*pd_save[1][5])/(x*pd_save[2][3]+y*pd_save[2][4]+z*pd_save[2][5]))** pd_lamb - light[0][1]/light[0][2]
    F[2] = ((x*pd_save[2][3]+y*pd_save[2][4]+z*pd_save[2][5])/(x*pd_save[0][3]+y*pd_save[0][4]+z*pd_save[0][5]))** pd_lamb - light[0][2]/light[0][0]
    F[3] = ((x*pd_save[2][3]+y*pd_save[2][4]+z*pd_save[2][5])/(x*pd_save[3][3]+y*pd_save[3][4]+z*pd_save[3][5]))** pd_lamb - light[0][2]/light[0][3]
    # for i in range(3):
    #     F[i] = z**led_lamb * (x*pd_save[i][3]+y*pd_save[i][4]+z*pd_save[i][5])**pd_lamb/(x**2+y**2+z**2)**(led_lamb/2+pd_lamb/2)
    # k =  (led_lamb+1)/(2*sympy.pi) *led_flux *pd_area
    # F[0] = k * z**led_lamb * (x*pd_save[0][3]+y*pd_save[0][4]+z*pd_save[0][5])**pd_lamb/(x**2+y**2+z**2)**(led_lamb/2+pd_lamb/2) - light[0][0]
    # F[1] = k * z**led_lamb * (x*pd_save[1][3]+y*pd_save[1][4]+z*pd_save[1][5])**pd_lamb/(x**2+y**2+z**2)**(led_lamb/2+pd_lamb/2) - light[0][1]
    # F[2] = k * z**led_lamb * (x*pd_save[2][3]+y*pd_save[2][4]+z*pd_save[2][5])**pd_lamb/(x**2+y**2+z**2)**(led_lamb/2+pd_lamb/2) - light[0][2]
    # F[3] = k * z**led_lamb * (x*pd_save[3][3]+y*pd_save[3][4]+z*pd_save[3][5])**pd_lamb/(x**2+y**2+z**2)**(led_lamb/2+pd_lamb/2) - light[0][3]
    return F
initguess = [1,1,1]
ans=  root(equations, initguess, method='lm')
print(ans.x)


# # theta = led out angle
# # phi = pd in angle
# # d = dis
# x,y,z= sympy.symbols('x,y,z')
# k = (led_lamb+1)/(2*sympy.pi) *led_flux *pd_area
# cons = light[0][0]/k
# # print(cons)
# # Pr = sympy.Eq( k * sympy.Pow((sympy.cos(theta)),led_lamb) * sympy.Pow((sympy.cos(phi)),pd_lamb) / (sympy.Pow(d,2)),light[0])
# # Pr = sympy.Eq( (costheta**led_lamb) * (cosphi**pd_lamb) / (d**2), cons)
# # print(pd_save)
# Ratio1 = sympy.Eq( ((x*pd_save[0][3]+y*pd_save[0][4]+z*pd_save[0][5])/(x*pd_save[1][3]+y*pd_save[1][4]+z*pd_save[1][5]))** pd_lamb,light[0][0]/light[0][1])
# Ratio2 = sympy.Eq( ((x*pd_save[1][3]+y*pd_save[1][4]+z*pd_save[1][5])/(x*pd_save[2][3]+y*pd_save[2][4]+z*pd_save[2][5]))** pd_lamb,light[0][1]/light[0][2])
# Ratio3 = sympy.Eq( ((x*pd_save[2][3]+y*pd_save[2][4]+z*pd_save[2][5])/(x*pd_save[0][3]+y*pd_save[0][4]+z*pd_save[0][5]))** pd_lamb,light[0][2]/light[0][0])

# print('before solve')
# # print(Ratio)
# ans = sympy.solve([Ratio1,Ratio2,Ratio3],(x,y,z))
# # ans = sympy.solveset([Ratio1,Ratio2,Ratio3],(x,y,z),domain=sympy.S.Reals)

# print(ans[1][0].is_real)

# print(ans)
# # sympy.nonlinsolve([x*y - 1, x + y - 1], (x, y))
# # print(sympy.sqrt(-9))
# # sympy.evaluate(sqrt(-sphi**2 - stheta**2 + 1))
# print(ans[0])
# print(ans[0][0])
# # fig1 = sympy.plotting.plot3d_parametric_surface((ans[0][0],ans[0][1],ans[0][2],(x,-3,3),(y,-3,3)))









fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')
for i in led_list:
    x,y,z = zip(i.point)
    u,v,w = zip(0.3*i.orien)
    ax.quiver(x,y,z,u,v,w,color='r')
    # ax.quiver(np.concatenate((i.point,i.point+i.orien),axis = 0),color = 'r')
for i in pd_list:
    x,y,z = zip(i.point)
    u,v,w = zip(0.3*i.orien)
    ax.quiver(x,y,z,u,v,w,color='g')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 0])
ax.set_title("config")

# plt.show()















# from scipy import signal
# import matplotlib.pyplot as plt
# fs = 1000.0  # Sample frequency (Hz)
# f0 = 300.0  # Frequency to be retained (Hz)
# Q = 30.0  # Quality factor
# # Design peak filter
# b, a = signal.iirpeak(f0, Q, fs)

# # Frequency response
# freq, h = signal.freqz(b, a, fs=fs)
# # Plot
# fig, ax = plt.subplots(2, 1, figsize=(8, 6))
# ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)))
# ax[0].set_title("Frequency Response")
# ax[0].set_ylabel("Amplitude (dB)")

# ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi)
# ax[1].set_ylabel("Angle (degrees)")
# ax[1].set_xlabel("Frequency (Hz)")

# plt.show()