# -*- coding: utf-8 -*-

from funcfile import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.cm as cm
import matplotlib.colors as colors

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

class hardware():
    def __init__(self, num):
        self.num = num
        self.ori_ang = np.tile(np.array([[0,0,1]]).T, (1,num)) # 3xpd
        self.ori_car = np.tile(np.array([[0,0,0]]).T, (1,num))
        self.pos = np.tile(np.array([[0,0,0]]).T, (1,num))# LED位置 [3xled_num]
    
    def ori_ang2cart(self, ori_ang):#ori_ang = 2xsensor_num np.array, 第一列傾角 第二列方位
        return np.stack((\
                    np.multiply(np.sin(ori_ang[0,:]), np.cos(ori_ang[1,:])),\
                    np.multiply(np.sin(ori_ang[0,:]), np.sin(ori_ang[1,:])),\
                    np.cos(ori_ang[0,:])    \
                        ),0)
    
    def set_config(self, config_num, alpha):
        if config_num ==0: # 放射狀
            beta = np.deg2rad(360/self.num)#方位角
            self.ori_ang = np.stack( (alpha * np.ones(self.num),(beta*np.arange(1, self.num+1))),0 )#2x?
            self.ori_car = ori_ang2cart(self.ori_ang) #3xpd
    
        
        elif config_num ==1: # 放射狀，一個朝中間

            beta = np.deg2rad(360/(self.num-1))#方位角
            self.ori_ang = np.stack(\
                                        ( alpha * np.ones(self.num-1),\
                                            (beta * np.arange(1,self.num))\
                                        ), 0)#2x?
            self.ori_ang = np.concatenate((self.ori_ang,np.array([[0,0]]).T),axis=1)
            self.ori_car = ori_ang2cart(self.ori_ang) #3xpd
        
        # elif config_num==2:
        #     a = (self.num*0.4)//1
        #     beta = np.deg2rad(360/a)#方位角
        #     ori_anga = np.stack(( \
        #                         alpha*np.ones(int(a)),\
        #                         (beta*np.arange(1,a+1))\
        #                         ))#2x?
        #     a = self.num - a
        #     beta = np.deg2rad(360 / a)#方位角
        #     ori_angb = np.stack( (\
        #                             3*pd_alpha*np.ones(int(a),),\
        #                             (beta * np.arange(1, a+1))\
        #                         ), 0)#2x?
        #     self.ori_ang = np.concatenate((ori_anga, ori_angb), axis=1)
        #     self.ori_car = ori_ang2cart(self.ori_ang) #3xpd
        

class led_coor_c(hardware):
    def __init__(self, num, hard_num, config_num, alpha):
        self.num = num
        hardware.__init__(self, num)
        self.set_hardware(hard_num)
        self.set_config(config_num, alpha)
      
    def set_hardware(self, hard_num):
        led_list = [\
                1.7*np.pi, 80*10**(-3), 1.35,1.15
                ]
        self.pt = led_list[hard_num]


class pd_coor_c(hardware):
    def __init__(self, num, hard_num, config_num, alpha):
        self.num = num
        hardware.__init__(self, num)
        self.set_hardware(hard_num)
        self.set_config(config_num, alpha)
        self.threashold = 10**(-9)
    def set_hardware(self, hard_num):
        pd_list = [\
                # pd_respon,pd_area,NEP,dark_current,shunt,capacitance
               [0.64, 6*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9, 740*10**(-12)],\
               [0.64, 5.7*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9, 680*10**(-12)],\
               [0.64, 33*10**(-6), 2*10**(-15), 50*10**(-12), 10*10**9, 4000*10**(-12)],\
               [0.64, 100*10**(-6), 2.8*10**(-15), 200*10**(-12), 5*10**9, 13000*10**(-12)],\
               [0.38, 36*10**(-6), 3.5*10**(-14), 100*10**(-12), 0.1*10**9, 700*10**(-12)]\
               ]
        self.respon = pd_list[hard_num][0]
        self.area = pd_list[hard_num][1]
        self.NEP = pd_list[hard_num][2]
        self.dark_current = pd_list[hard_num][3]
        self.shunt = pd_list[hard_num][4]
        self.capacitance = pd_list[hard_num][5]

        


class test_point_c():
    def __init__(self, scenario):
        match scenario:
            case 0: 
                #平移樣本
                self.testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 0:3:10j].reshape((3,-1)) # 3xkp
                #旋轉樣本
                self.testp_rot  = ((np.mgrid[0:0:1j, 10:60:6j, 0:360:11j])[:,:,:,:-1])
                self.testp_rot = np.deg2rad(self.testp_rot.reshape((3, -1)))
                self.testp_rot = np.concatenate((self.testp_rot, np.array([[0,0,0]]).T ), axis=1)\
                                    +np.array([[np.pi,0,0]]).T # 加上一個不多做旋轉的樣本點，並將所有樣本點pitch轉pi變面對面
            # 平面上
            case 1:
                self.testp_pos = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j, 2.5:2.5:1j].reshape((3,-1)) # 3x?
                self.testp_rot = np.array([[np.pi,0,0]]).T
                # print(testp_pos[0,:].shape)
            
            # 球狀空間：任意位置與角度皆能求解
            case 2:
                sample = 6
                # 平移樣本以球座標系表示的距離項
                dis_sample = np.linspace(0, 3, 4+1)[1:]

                # 用球座標系的天頂角v與方位角 表示任意姿態任意方位（不含距離）
                u, v = np.meshgrid(\
                                    np.linspace(0, 2*np.pi, 2*sample+1)[0:-1:1],\
                                    np.linspace(0, np.pi, sample+1)[1:-1:1]\
                                )
                # 加上在極北點的角度
                u = np.append(u.reshape((-1,)), 0)
                v = np.append(v.reshape((-1,)), 0)
                # 加上在極南點的角度
                u = np.append(u.reshape((-1,)), 0)
                v = np.append(v.reshape((-1,)), np.pi)
                # 把球座標系換到卡氏
                x = (1 * np.cos(u) * np.sin(v))
                y = (1 * np.sin(u) * np.sin(v))
                z =( 1 * np.cos(v))
                # U:xyz合併成矩陣[3x?]
                U = np.stack((x,y,z))
                # 將不同距離考慮進來
                U = np.tile(U, (dis_sample.size, 1, 1)).transpose((1,2,0))
                self.testp_pos = np.multiply(dis_sample.reshape((1,1,-1)), U)
                # reshape成2D [3x?]的大小
                self.testp_pos = self.testp_pos.reshape((3,-1))
                
                # 旋轉樣本點：pitch=0, roll和yaw以任意姿態表示
                self.testp_rot = np.stack((np.zeros(u.shape),v,u))
            
            # scenario:0 的擴大版
            case 3:
                ma = 10 # 空間邊長
                self.testp_pos = np.mgrid[-ma/2:ma/2:10j, -ma/2:ma/2:10j, 0:ma:10j].reshape((3,-1)) # 3x?
                self.testp_rot = ((np.mgrid[0:0:1j, 10:60:6j, 0:360:11j])[:,:,:,:-1])
                self.testp_rot = np.deg2rad(self.testp_rot.reshape((3,-1)))
                self.testp_rot = np.concatenate((self.testp_rot,np.array([[0,0,0]]).T ),axis=1)\
                            +np.array([[np.pi,0,0]]).T
            case 4: # 1 to 1
                self.testp_pos = np.array([[1,1,1.5]]).T
                self.testp_rot = testp_rot = np.array([[np.pi,0,0]]).T
        self.kpos = self.testp_pos.shape[1] # 平移樣本數量
        self.krot = self.testp_rot.shape[1] # 旋轉樣本數量





# led_coor = led_coor_c()
pd_coor = pd_coor_c(3, 0, 0, 30)
# pd_coor.set_config(0,30)
print(pd_coor.NEP)
print(pd_coor.ori_car)
print(pd_coor.ori_ang)
