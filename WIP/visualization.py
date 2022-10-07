import vpython
from vpython import *
import numpy as np
import pickle as pk

#x1, y1, z1, x2, y2, z2 = np.load('..\\data\\3Dpen.npy')
# ball1 = vpython.sphere(color = color.green, radius = 0.3, make_trail=True, retain=20)
# ball2 = vpython.sphere(color = color.blue, radius = 0.3, make_trail=True, retain=20)
# rod1 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.05)
# rod2 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.05)
# base  = box(pos=vector(0,-4.25,0),axis=vector(1,0,0),
#             size=vector(10,0.5,10) )
# s1 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))
# s2 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))




#with open("data.pkl", "rb") as f:
with open("exp/optimal_traj_data.pkl", "rb") as f:
    vals = pk.load(f)
#vals = np.load("data.npy", allow_pickle=True) 

# print(vals)
# input()

x = vals[0]
y = vals[1]
theta = vals[2]

alpha = vals[3]
phi_l = vals[4]
phi_r = vals[5]

v_alpha = vals[6]
v_phi_l = vals[7]
v_phi_r = vals[8]

rw = 33.1*10e-3      # wheels radius
dw = 49*10e-3      # semidistance wheels
mw = 0.028      # mass wheel
mb = 0.277      # pend mass
b = 48.67*10e-3       # pend Length

print(dw)

pend_tip = vpython.sphere(color = color.green, radius = 0.09)#, make_trail=True, retain=20)
pend = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.05)
body = box(pos=vector(0,rw,0), axis=vector(1,0,0), size=vector(dw,rw,2*dw), color = color.red)#, make_trail=True)

#r_wheel = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=rw, color=color.blue, make_trail=True)
pos0 = vector(x[0]-np.cos(theta[0]+np.pi/2)*dw, rw, y[0]-np.sin(theta[0]+np.pi/2)*dw)
axis0 = -(body.pos-pos0)*0.2
r_wheel = cylinder(pos=pos0,axis=axis0, radius=rw, color=color.blue, make_trail=True)

#l_wheel = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=rw, color=color.blue, make_trail=True)
pos0 = vector(x[0]+np.cos(theta[0]+np.pi/2)*dw, rw, y[0]+np.sin(theta[0]+np.pi/2)*dw)
axis0 = -(body.pos - pos0)*0.2 
l_wheel = cylinder(pos=pos0, axis=axis0, radius=rw, color=color.blue, make_trail=True)

base  = box(pos=vector(0,-0.5,0),axis=vector(1,0,0), size=vector(100,0.5,100))


input()
print('Start')
i = 0
while i<len(x):
    rate(30)

    #i = i % len(x)
    print(i)
    scene.camera.pos =  vector(x[i], 5, y[i]+5)
    scene.camera.axis = vector(0, -5, -5)
    body.pos = vector(x[i], rw, y[i])
    body.axis = vector(np.cos(theta[i]),0,np.sin(theta[i]))
    body.size = vector(dw,rw,2*dw)

    r_wheel.pos = vector(x[i]-np.cos(theta[i]+np.pi/2)*dw, rw, y[i]-np.sin(theta[i]+np.pi/2)*dw)
    r_wheel.axis = -(body.pos-r_wheel.pos)*0.2

    l_wheel.pos = vector(x[i]+np.cos(theta[i]+np.pi/2)*dw, rw, y[i]+np.sin(theta[i]+np.pi/2)*dw)
    l_wheel.axis = -(body.pos - l_wheel.pos)*0.2 #vector(-np.cos(theta[i]+np.pi/2)*dw,0,-np.sin(theta[i]+np.pi/2)*dw)*0.5


    pend.pos = vector(x[i], rw+rw/2, y[i])
    pend.axis = vector(np.sin(alpha[i])*np.cos(theta[i]), np.cos(alpha[i]), np.sin(alpha[i])*np.sin(theta[i]))*b
    #print(alpha[i])
    #print((np.cos(alpha[i])*np.cos(theta[i]), np.sin(alpha[i]), np.cos(alpha[i])*np.sin(theta[i])))
    pend_tip.pos = vector(x[i]+b*np.sin(alpha[i])*np.cos(theta[i]), rw+rw/2 + b*np.cos(alpha[i]), y[i] + b*np.sin(alpha[i])*np.sin(theta[i]))
    
    #input()
    # ball1.pos = vector(x1[i], z1[i], y1[i])
    # ball2.pos = vector(x2[i], z2[i], y2[i])
    # rod1.axis = vector(x1[i], z1[i], y1[i])
    # rod2.pos = vector(x1[i], z1[i], y1[i])
    # rod2.axis = vector(x2[i]-x1[i], z2[i]-z1[i], y2[i]-y1[i])
    # s1.pos = vector(x1[i], -3.99, y1[i])
    # s2.pos = vector(x2[i], -3.99, y2[i])
    i = i + 1