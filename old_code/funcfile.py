import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

# -------------------------- function def ------------------------------------


'''座標轉換 matrix '''
def ori_ang2cart(ori_ang):#ori_ang = 2xsensor_num np.array, 第一列傾角 第二列方位
    return np.stack((\
    np.multiply(np.sin(ori_ang[0,:]), np.cos(ori_ang[1,:])),\
    np.multiply(np.sin(ori_ang[0,:]), np.sin(ori_ang[1,:])),\
    np.cos(ori_ang[0,:])    \
        ),0)

# 用法：rotate_x(pos,np.deg2rad(45))
def rotate_x(ang): #ang[rad](3*3)
    rot = np.array([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])
    # print(rot)
    return rot #是一個matrix
def rotate_y(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad](3*3)
    rot = np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
    # print(rot)
    return rot #是一個matrix
def rotate_z(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad](3*3)
    rot = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
    # print(rot)
    return rot #是一個matrix
def rotate_mat(ang_list):#ang_list[x,y,z]的角度in rad #順序是先轉x->y->z
    return np.dot( rotate_z(ang_list[2]),(np.dot(rotate_y(ang_list[1]),rotate_x(ang_list[0]))) ) 
def rotate(mat, ang_list):#mat 3x? #ang_list [a,b,c]
    return np.dot(rotate_mat(ang_list),mat)
def trans(mat,trans):
    if len(mat.shape)==2: #是二維矩陣
        (_,num)=mat.shape
        # print(mat)
        # print(np.transpose(np.tile(trans,(num,1))))
        return mat + np.tile(trans,(1,num))
    else: #bug
        print('translation error: 你可能忘了把轉移向量變成[[]]')
        return None

def homogeneous_mat(ang_list, trans): #trans(3,1)
    hom = np.zeros((4,4))
    hom[:3,:3]= rotate(ang_list)
    hom[3,3]=1
    hom[:3,3]=trans
    return hom
def inv_hom_mat(ang_list, trans):#trans(3,1)
    hom = np.zeros((4,4))
    hom[:3,:3]= np.transpose(rotate(ang_list))
    hom[3,3]=1
    hom[:3,3]=-1*np.dot(hom[:3,:3],trans)
    return hom

'''coordinate transfer'''
# 產生k2個rotation matrix
# shape(k2,3,3) 每個3x3是一個rotation matrix
def testp_rot_matlist(testp_rot): # testp_rot [[rotx][roty][rotz]] 3x?
    krot = testp_rot.shape[1]
    out = np.zeros((krot,3,3)) # shape(k2,3,3)
    for i in range(krot):
        out[i,:,:] = rotate_mat(testp_rot[:,i])
    return out ## shape(k2,3,3) 每個3x3是一個rotation matrix

def testp_euler_matlist(testp_rot): # testp_rot [[roll][pitch][yaw]] 3x?
    krot = testp_rot.shape[1]
    out = np.zeros((krot,3,3)) # shape(k2,3,3)
    for i in range(krot):
        out[i,:,:] = rotate_mat(testp_rot[:,i])
    return out ## shape(k2,3,3) 每個3x3是一個rotation matrix

# 把多個點轉到global coor上
# pos是多個點 3x?
# testp_rot是[[rotx][roty][rotz]] ，多個轉換參數
# return krotx3xm
def global_testp_after_rot(pos, testp_rot): #pos(or ori)[3x?] #testp_rot (krot,3)
    rot_list = testp_rot_matlist(testp_rot) # (krot,3,3)
    out = rot_list @ pos
    return out # krotx3xm

# out(krot,kpos,3,m) 
def global_testp_trans(pos , testp_pos): 
    # pos [krotx3xm] or [3xm]
    # testp_pos [3xkpos]
    if len(pos.shape)==3:
        kpos = testp_pos[1].size # kpos
        krot =  pos.shape[0] #krot
        m =  pos.shape[2] #led_num

        out = np.tile(pos,(kpos,1,1,1)).transpose((0,1,3,2))+\
              np.tile(testp_pos.T,(krot,pos.shape[2],1,1)).transpose((2,0,1,3))
        return out # kpos krot m 3
    elif len(pos.shape)==2:
        print('error in global_testp_trans')


'''計算d,in_ang,out_ang'''
def cal_d_in_out(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car,krot,kpos,led_num,pd_num):
    #glob_led_pos [krot kpos 3 led_num]
    #glob_led_ori [krot kpos 3 led_num]
    #pd_pos [3 pd_num]
    #pd_ori_car [3 pd_num]
    #dis = np.zeros((krot,kpos,led_num,pd_num))
    in_ang = np.zeros((krot,kpos,led_num,pd_num))
    out_ang = np.zeros((krot,kpos,led_num,pd_num))

    #pos_delta = np.zeros((krot,kpos,led_num,pd_num,3)) #led-pd: pd pointint to led
    pos_delta = np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose((1,2,4,0,3)) \
        - np.tile(pd_pos,(krot,kpos,led_num,1,1)).transpose((0,1,2,4,3))
    dis = np.sqrt(np.sum(np.square(pos_delta),axis=4)) # krot,kpos,led_num,pd_num
    in_ang = np.inner(pos_delta,pd_ori_car.T)
    for pd in range(pd_num):
        for led in range(led_num):
            # (x-x)^2 sqrt
            # glob_led_pos[:,:,:,led]  krotxkposx3
            pd_extend = np.tile(pd_pos,(krot,kpos,1,1))
            pos_delta[:,:,led,pd,:] = glob_led_pos[:,:,:,led]-pd_extend[:,:,:,pd]#krotxkposx3
            dis[:,:,led,pd] = np.sqrt(np.square(pos_delta[:,:,led,pd,:]).sum(axis=2))
            # in_ang[:,:,led,pd]= 

    for pd in range(pd_num):
        # 計算該pd與all testpoints的角度
        in_ang[:,:,:,pd] = pos_delta[:,:,:,pd,0]*pd_ori_car[0,pd]+pos_delta[:,:,:,pd,1]*pd_ori_car[1,pd]+pos_delta[:,:,:,pd,2]*pd_ori_car[2,pd]

        # krot kpos led pd  #glob_led_ori(krot,3,led_num) 
        # out_ang[:,:,:,pd] = -pos_delta[:,:,:,pd,0]* glob_led_ori[,0,] - pos_delta[:,:,:,pd,1]*   -pos_delta[:,:,:,pd,2]*
    for led in range(led_num):
        for r in range(krot):
            out_ang[r,:,led,:] = -pos_delta[r,:,led,:,0]*glob_led_ori[r,0,led] - pos_delta[r,:,led,:,1]*glob_led_ori[r,1,led] - pos_delta[r,:,led,:,2]*glob_led_ori[r,2,led]
    in_ang = np.arccos(np.divide(in_ang,dis))
    out_ang = np.arccos(np.divide(out_ang,dis))
    
    return dis,in_ang,out_ang




'''
def lamb_order(semi): #semi power angle in degree
    return -1*np.log(2)/np.log(np.cos(np.deg2rad(semi)))
# print(lamb_order(28))
'''

'''
def cal_ori(light_f,obs_m,obs_num,obs_ori):
    pd_num = obs_num
    pd_m = obs_m
    pd_ori_car = obs_ori
    led_usable = np.sum(~np.isnan(light_f),axis=1)>2 #led,
    light_led = light_f[led_usable,:] #ledu, pd
    
    # =============================================================================
    # 取強度最大者作為ref1，建立平面的基準
    # 並利用mask將light_led分成ref和other
    # => 計算ratio
    # =============================================================================
    ref1 = np.nanargmax(light_led, axis = 1) #ledu,
    mask = np.full(light_led.shape, False)
    mask[np.arange(led_usable.sum()),ref1] = True

    data_ref = light_led[mask].reshape(-1,1)#ledu 1
    data_other = light_led[~mask].reshape(ref1.sum(),-1)# ledu other
    # ref/other
    ratio = np.power(np.divide(data_ref, data_other),1/pd_m) #led_u x other
    # in_ang  krot,kpos,led_num,pd_num

    # =============================================================================
    # 計算平面normal vector[ledu other 3]
    # =============================================================================
    #ledu x other x 3
    nor = np.tile(pd_ori_car.T,(led_usable.sum(),1,1))[np.tile(mask,(3,1,1)).transpose(1,2,0)].reshape(led_usable.sum(),1,3)\
        - np.multiply(\
                    np.tile(pd_ori_car.T,(led_usable.sum(),1,1))[np.tile(~mask,(3,1,1)).transpose(1,2,0)].reshape(led_usable.sum(),-1,3)\
                    ,ratio.reshape(led_usable.sum(),-1,1))
# =============================================================================
#     check_dot = (np.inner(np.array([[0,1,1]]),nor))
#     check_dot = np.sum(~(np.isclose(check_dot,np.zeros((3,6)))|np.isnan(check_dot)))
#     print('-----------------------')
#     print('False normal vector:' ,check_dot)
#     print('-----------------------')
# =============================================================================
    # =============================================================================
    # 取data_other強度最大者作為ref2，當cross的基準
    # 並利用mask2將data other分兩半
    # => 計算cross
    # =============================================================================
    ref2 = np.nanargmax(data_other, axis = 1)
    mask2 = np.full(data_other.shape, False)
    mask2[np.arange(led_usable.sum()),ref2] = True

    nor_ref = nor[mask2].reshape(-1,1,3) #ledu,1,3
    nor_other = nor[~mask2].reshape(led_usable.sum(),-1,3) #ledu,other-1,3
    cross = np.cross(np.tile(nor_ref,(1,pd_num-2,1)),nor_other)#ledu,other-1,3
    cross = np.divide(cross, np.tile(np.sqrt(np.sum(np.square(cross),axis=2)),(1,1,1)).transpose(1,2,0))#ledu,other-1,3
    cross_pstv = np.tile(pd_ori_car.T,(led_usable.sum(),1,1))  [np.tile(mask,(3,1,1)).transpose(1,2,0)].reshape((-1,3)) #ledu 3
    cross_mask = np.sum(np.multiply(cross, np.tile(cross_pstv,(pd_num-2,1,1)).transpose(1,0,2)),axis=2)<0#ledu other-1
    cross = np.where(np.tile(cross_mask,(3,1,1)).transpose(1,2,0),-cross,cross)
    #cross [ledu other-1 3]


# =============================================================================
#     check_cross = (np.sum(np.multiply(cross,np.tile(testp_pos.T/np.sqrt(np.sum(np.square(testp_pos))),(led_usable.sum(),pd_num-2,1))),axis=2))
#     check_cross = np.sum(~(np.isnan(check_cross) | np.isclose(check_cross,np.ones((led_usable.sum(),pd_num-2)))))
#     print('-----------------------')
#     print('False cross vector:' ,check_cross)
#     print('-----------------------')
# =============================================================================



def ang_from_ori(a_ang,b_ang):#[2xa] [2xb]
    # a_angg = np.transpose(np.array([a_ang.T]*b_ang.shape[1]),(1,0,2))
    a = np.tile(a_ang,(b_ang.shape[1],1,1)).transpose(1,2,0)
    # print(a_ang.shape,'a')
    # b_angg = np.array([b_ang.T]*a_ang.shape[0])
    b = np.tile(b_ang,(a_ang.shape[1],1,1)).transpose(1,0,2)
    # print(b_ang.shape,'a')
    out = np.arccos(\
            np.multiply(np.multiply(np.sin(a[0,:,:]),np.sin(b[0,:,:])), np.cos(a[1,:,:]-b[1,:,:]))+\
            np.multiply(np.cos(a[0,:,:]),np.cos(b[0,:,:])))
    # out = np.arccos(  np.multiply( np.multiply\
    #     (np.sin(a_angg[:,:,0]),np.sin(b_angg[:,:,0])), np.cos(a_angg[:,:,1]-b_angg[:,:,1]))  + \
    #     np.multiply(np.cos(a_angg[:,:,0]), np.cos(b_angg[:,:,0]))  )
    # print(out.shape,'out')
    return out #[axb]
'''
# def ang_btw(a1,b1,a2,b2): #alpha1,beta1,alpha2,beta2
#     a1,b1,a2,b2 = np.deg2rad(np.array([a1,b1,a2,b2]))
#     return np.arccos(np.sin(a1)*np.sin(a2)*np.cos(b1-b2)+np.cos(a1)*np.cos(a2))
'''
def stereo_sph2pol(ori):#ori[2x?]
    new = np.zeros(ori.shape)
    new[0,:] = np.divide(np.sin(ori[0,:]), 1+np.cos(ori[0,:]) )
    new[1,:] = ori[1,:]
    return new #[2x?]
def stereo_pol2sph(pol): #pol:R,ang
    out = np.zeros(pol.shape)
    out[0,:] = 2 * np.arctan(pol[0,:])
    out[1,:] = pol[1,:]
    return out                     
def stereo_3dto2d(p3d):#p3d[3x?]
    p2d = np.divide( p3d[:2,:] , (1+p3d[2,:]) )
    return p2d
def stereo_2dto3d(p2d): #p2d[2x?]
    out = np.stack( (2*p2d[0,:], 2*p2d[1,:], 1-np.sum(np.square(p2d),axis=0)) , 0 ) #[3x?]
    return np.divide(out, 1+ np.sum(np.square(p2d),axis=0) ) #[3x?]
def pol2cart(pol): #pol(R,ang) [2x?]
    return np.multiply(np.stack( ( np.cos(pol[1,:]), np.sin(pol[1,:])  ),0 ), pol[0,:] )

def rodrigue(k_vec1,ang): #k:[3]
    k_vec = k_vec1.reshape((3,))
    k = (1/np.sqrt(np.sum(np.square(k_vec))))*k_vec
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    I = np.eye(3)
    R = I + np.sin(ang)*K + (1-np.cos(ang))*(np.matmul(K,K)) #3x3
    return R #3x3
def rodrigue_1mul(k_vec1,ang): #k:[3]
    k_vec = k_vec1.reshape((3,))
    k = (1/np.sqrt(np.sum(np.square(k_vec))))*k_vec
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    I = np.eye(3)
    R = np.tile(I,(ang.size,1,1)) \
        + np.multiply(np.sin(ang).reshape(-1,1,1),np.tile(K,(ang.size,1,1))) \
        + np.multiply((1-np.cos(ang)).reshape(-1,1,1),np.tile((np.matmul(K,K)),(ang.size,1,1))) #angx3x3
    return R #angx3x3

def rodrigue_mulmul(k_vec,ang): #k:[sample,3] ang[sample,]
    k = np.multiply((1/np.sqrt(np.sum(np.square(k_vec),axis=1))).reshape((-1,1)),k_vec) #sample,
    K = np.zeros((ang.size,3,3))#np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]]) #sample,3,3
    K[:,0,1] = -k[:,2]
    K[:,0,2] = k[:,1]
    K[:,1,0] = k[:,2]
    K[:,1,2] = -k[:,0]
    K[:,2,0] = -k[:,1]
    K[:,2,1] = k[:,0]
    I = np.eye(3)
    R = np.tile(I,(ang.size,1,1)) \
        + np.multiply(np.sin(ang).reshape(-1,1,1),K) \
        + np.multiply((1-np.cos(ang)).reshape(-1,1,1),(np.matmul(K,K))) #angx3x3
    return R #samplex3x3
'''
def cart2sph(cart_v):#3x?
    cart = np.divide(cart_v, np.sqrt(np.sum(np.square(cart_v),axis=0).reshape((1,-1))))
    return np.array([   np.arccos(cart[2,:])   ,  np.arctan(np.divide(cart[1,:],cart[0,:]))- (cart[0,:]<0)*(np.pi)  ])# np.divide( np.arctan(cart[1,:]), cart[0,:]) - (cart[0,:]<0)*(np.pi)   ])#2x?


'''
def rotate_y_mul(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad] list 1x?
    rot = np.zeros((ang.size,3,3))
    rot[:,0,0] = np.cos(ang)
    rot[:,0,2] = np.sin(ang)
    rot[:,1,1] = np.ones((ang.size,))
    rot[:,2,0] = -np.sin(ang)
    rot[:,2,2] = np.cos(ang)
    # np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
    # print(rot)
    return rot #是一個matrix
def rotate_z_mul(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad] list 1x?
    rot = np.zeros((ang.size,3,3))
    rot[:,0,0] = np.cos(ang)
    rot[:,0,1] = -np.sin(ang)
    rot[:,1,0] = np.sin(ang)
    rot[:,1,1] = np.cos(ang)
    rot[:,2,2] = np.ones((ang.size,))
    #rot = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
    # print(rot)
    return rot #是一個matrix
'''

def interactive_btw_pdled(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car):
    (kpos,krot,led_num,_) = glob_led_pos.shape
    pd_num = pd_pos.shape[1]
    
    pos_delta = np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose((1,2,3,0,4)) \
        - np.tile(pd_pos.T,(kpos,krot,led_num,1,1))
    
    dis = np.sqrt(np.sum(np.square(pos_delta),axis=4)) # krot,kpos,led_num,pd_num
    #print(dis)
    in_ang = np.arccos(np.divide(np.sum(np.multiply( np.tile(pd_ori_car.T,(kpos,krot,led_num,1,1)), pos_delta), axis=4), dis))
    out_ang = np.arccos( np.divide(np.sum(np.multiply(    -pos_delta,np.tile(glob_led_ori,(pd_num,1,1,1,1)).transpose((1,2,3,0,4))   ),axis=4), dis ) )

    # krot,kpos,led_num,pd_num
    return dis,in_ang,out_ang


def filter_view_angle(mat,ang):
    mat_view = np.empty_like(mat)
    mat_view[:] = mat
    mat_view[mat_view >= ang] = np.nan
    return mat_view



# 計算平面法向量
def get_surface(light_led,light_pd,led_num,pd_num,kpos,krot,led_m,pd_m,led_ori_car,pd_ori_car):
# print('Led, Pd usable amount: ',ledu,pdu)
    # 光最強的那個當reference
    ref1_led = np.nanargmax(light_led, axis = 3) #kp,kr,ledu,
    ref1_pd = np.nanargmax(light_pd, axis = 2) #kp,kr,pdu,

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
    
    # data_ref: ref1數據
    # data_other: 除了ref1的數據
    led_data_ref = light_led.copy()
    led_data_ref .mask = (light_led .mask | ~maskled)
    led_data_ref = np.sort(led_data_ref,axis=3)[:,:,:,0].reshape(kpos,krot,led_num,1)
    led_data_other = light_led.copy()
    led_data_other.mask = (light_led.mask | maskled)
    
    pd_data_ref = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
    pd_data_ref.mask = (light_pd.mask | ~maskpd)
    pd_data_ref = np.sort(pd_data_ref,axis=2)[:,:,0,:].reshape(kpos,krot,1,pd_num)
    pd_data_other = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
    pd_data_other.mask = (light_pd .mask | maskpd)




    # 取ratio: ref/other
    ratio_led = np.power(np.divide(led_data_ref, led_data_other),1/pd_m)
    ratio_pd = np.power(np.divide(pd_data_ref, pd_data_other),1/led_m) #other, pdu
    # in_ang  krot,kpos,led_num,pd_num
    


    # =============================================================================
    # 計算平面normal vector[ledu other 3]
    # =============================================================================
    # 將硬體指向根據ref與other分類
    #kpos x krot x ledu x other x 3
    conf_led = np.tile(pd_ori_car.T,(kpos,krot,led_num,1,1))
    # print(conf_led,'conf')
    conf_led_ref = np.sort( (np.ma.masked_array(conf_led,np.tile((light_led.mask | ~maskled),(3,1,1,1,1)).transpose(1,2,3,4,0))),axis=3)[:,:,:,0,:].reshape(kpos,krot,led_num,1,3)
    conf_led_other = np.ma.masked_array(conf_led,np.tile(led_data_other.mask,(3,1,1,1,1)).transpose(1,2,3,4,0))
    # print(conf_led_other,'other')
    
    # 計算normal vector
    nor_led = conf_led_ref - np.multiply(ratio_led.reshape(kpos,krot,led_num,-1,1),conf_led_other)

    # 將硬體指向根據ref與other分類
    # kp kr l p 3
    conf_pd = np.tile(led_ori_car,(kpos,krot,pd_num,1,1)).transpose(0,1,4,2,3) # kp kr l p
    # kp kr 1 p 3
    conf_pd_ref = np.sort( (np.ma.masked_array(conf_pd,np.tile((light_pd.mask | ~maskpd),(3,1,1,1,1)).transpose(1,2,3,4,0))),axis=2)[:,:,0,:,:].reshape(kpos,krot,1,-1,3)
    # kp kr l p 3
    conf_pd_other = np.ma.masked_array(conf_pd,np.tile(pd_data_other.mask,(3,1,1,1,1)).transpose(1,2,3,4,0))
   
    # 計算normal vector
    # kp kr l p
    nor_pd = conf_pd_ref - np.multiply(ratio_pd.reshape(kpos,krot,led_num,-1,1),conf_pd_other)
    return nor_led,nor_pd,conf_led_ref,conf_pd_ref,led_data_other,pd_data_other


def get_cross(led_data_other,pd_data_other,light_led,light_pd,led_num,pd_num,kpos,krot,nor_led,nor_pd,conf_led_ref,conf_pd_ref):
    # =============================================================================
    # 取led_data_other強度最大者作為ref2_led，當cross的基準(也就是全部裡面第二大)
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
    
    # 將normal vector根據ref2與other分兩半
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

# =============================================================================
#     # 驗算cross led
#     check_cross_led = np.sum(np.multiply(cross_led,(np.tile( \
#                                                             np.divide(testp_pos,\
#                                                                       np.tile(np.sqrt(np.sum(np.square(testp_pos),axis=0)),(1,1))\
#                                                                       ),\
#                                                             (krot,led_num,pd_num,1,1)).transpose(4,0,1,2,3))\
#                                          ),axis=4)#kp kr l p
#     check_cross_led = np.isclose(np.ma.masked_invalid(check_cross_led),np.ones(check_cross_led.shape))
#     check_cross_led_sum = np.sum(~check_cross_led)
# =============================================================================

    
    # kp kr l p 3
    cross_pd = np.ma.masked_array(np.cross(np.tile(nor_pd_ref,(1,1,led_num,1,1)),nor_pd_other)\
                                   ,nor_pd_other.mask )#ledu,other-1,3
    
    cross_pd = np.divide(cross_pd, np.tile(np.sqrt(np.sum(np.square(cross_pd),axis=4)),(1,1,1,1,1)).transpose(1,2,3,4,0))#ledu,other-1,3
    
    cross_pd_mask = np.sum(np.multiply(conf_pd_ref, cross_pd),axis=4)<0 ## kp kr l p
    cross_pd = np.ma.masked_array(np.where(np.tile(cross_pd_mask,(3,1,1,1,1)).transpose(1,2,3,4,0),-cross_pd,cross_pd),\
                                   nor_pd_other.mask)
    return cross_led,cross_pd