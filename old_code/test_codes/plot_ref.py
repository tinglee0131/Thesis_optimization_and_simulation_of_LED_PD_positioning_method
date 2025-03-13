# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import funcfile_old as func
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 


def plot_1D(y):
    plt.plot(np.arange(y.size)+1,y)
    plt.show()

def plot_2D(x,y):
    plt.plot(x,y)
    plt.show()


def pickLEDandPD(LED,PD):
    distance = np.array([1, 2, 3, 4, 5])



def led_output():
    pass

if __name__ == '__main__':

    power = 100
    theta = np.linspace(-45,45,200)
    dis = np.linspace(3,5,20)
    thetaa, diss = np.meshgrid(theta,dis)
    leng = thetaa.shape
    
    psi = np.zeros(leng)
    # print(thetaa.shape,diss.shape,psi.shape)
    pd = np.ones(3)
    led = np.array([func.lamb_order(10),1])



    p1 = func.src2pdcurrent(thetaa,psi,diss,pd,led)


    fig = plt.figure()
    ax3d = plt.axes(projection="3d")
    
    ax3d = plt.axes(projection='3d')
    ax3d.plot_surface(thetaa,diss,p1, cmap='plasma')
    ax3d.set_title('Radiant Flux at different distance and angle')
    ax3d.set_xlabel('LED出射角')
    ax3d.set_ylabel('距離')
    ax3d.set_zlabel('Relative radiant flux(%)')
    plt.show()





    # sz = 100
    # x = np.linspace(-5, 5, sz)
    # y = np.linspace(-5, 5, sz)
    # xx, yy = np.meshgrid(x, y)
    # z = np.zeros((sz, sz))
    # nx = np.zeros((sz, sz))
    # ny = np.zeros((sz, sz))
    # nz = np.ones((sz, sz))
    
    # src = np.array([0, 0, 10,0,0.5,-1])
    # pd_para =[1,0.1,1] #pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]
    # led_para = [1,100]
    
    # dis = func.ptcloud_src_dis(src,xx,yy,z)
    # outang = func.outangle(src,xx,yy,z)
    # inang = func.inangleWsrc(src,xx,yy,z,nx,ny,nz)
    # pd = func.src2pdcurrent(inang,outang,dis,pd_para,led_para)
    
    # fig = plt.figure(figsize=(8, 6))
    # ax3d = plt.axes(projection="3d")
    
    # ax3d = plt.axes(projection='3d')
    # ax3d.plot_surface(xx,yy,pd, cmap='plasma')
    # ax3d.set_title('Illuminance at different place')
    # ax3d.set_xlabel('X')
    # ax3d.set_ylabel('Y')
    # ax3d.set_zlabel('Illuminance')
    # plt.show()
