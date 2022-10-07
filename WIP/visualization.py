import vpython
from vpython import *
import numpy as np
import pickle as pk





with open("experiments/optimal_traj_data2.pkl", "rb") as f:
    vals = pk.load(f)

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


input("Press Enter to Start the visualization")
print('Start')
i = 0
while i<len(x):
    rate(30)

    #i = i % len(x)
    print(f"step {i}")
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
    
    i = i + 1