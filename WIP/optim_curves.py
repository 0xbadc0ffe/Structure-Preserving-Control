from re import U
from tkinter import FALSE
import numpy as np
import dyn_sys as dyn
import matplotlib.pyplot as plt
import os
from scipy.linalg import expm, sinm, cosm
import pickle as pk
import cyipopt as ipopt
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from cyipopt import minimize_ipopt
from scipy.optimize._numdiff import approx_derivative





################### UTILS

def flat(arr):
    return np.asfarray(arr).flatten()

def trasp(arr):
    return np.asmatrix(arr).T

def my_squeeze(mat):
    return np.array(mat).ravel()

# Da polare (angolo zero all'eq stabile) a cartesiano
def pol2cart(theta, r=None):
    if r is None:
        return np.stack((L*np.sin(theta), -L*np.cos(theta))).reshape(1,2)
    else:
        return np.stack((r*np.sin(theta), -r*np.cos(theta))).reshape(1,2)



# Step della dinamica trvoata con il variational integrator
def varIntWIP(t, q, u=None, stop_on_complex_root=False):
    gk = trasp(q[:3])   #(x,y,theta)
    sk = trasp(q[3:6])
    vk = trasp(q[6:9])

    gravity = 9.81
    rw = 33.1*10e-3      # wheels radius
    dw = 49*10e-3      # semidistance wheels
    mw = 0.028      # mass wheel
    mb = 0.277      # pend mass
    b = 48.67*10e-3       # pend Length
    h = eps
    IW = np.matrix([[1,0,0],[0, 7411*10e-6,0],[0,0, 4957*10e-6]])
    IB = np.matrix([[543108*10e-6,0,0],[0, 481457*10e-6,0],[0,0, 153951*10e-6]])
    IWyy = IW[1,1]
    IWzz = IW[2,2]
    IBxx = IB[0,0]
    IByy = IB[1,1]
    IBzz = IB[2,2]

    if u is None:
        u = np.array([0,0,0])
    else:
        u = np.array([0,u[0],u[1]])


    A = np.matrix([[0, -rw, -rw],[0,0,0],[0,rw/dw,-rw/dw]])

    V = np.matrix([[0,-vk[0], vk[1]], [vk[0], 0, vk[2]], [0,0,0]])
    #gk_next = (gk.T@expm(-h*A@V)).T

    # G = np.matrix([[np.cos(gk[2]), -np.sin(gk[2]), gk[0]], [np.sin(gk[2]), np.cos(gk[2]), gk[1]], [0,0,1]])
    # G_n = G@expm(-h*A@V)
    # gk_next = trasp(np.array([G_n[0,2],G_n[1,2], np.arccos(G_n[0,0])]).flatten())


    Av = A@vk
    Avm = np.matrix([[0,-Av[2], Av[0]], [Av[2], 0, Av[1]], [0,0,0]])
    G = np.matrix([[np.cos(gk[2]), -np.sin(gk[2]), gk[0]], [np.sin(gk[2]), np.cos(gk[2]), gk[1]], [0,0,1]])
    G_n = G@expm(-h*Avm)
    ang = np.arctan2(G_n[1,0],G_n[0,0])    # np.arccos(G_n[0,0])
    gk_next = trasp(np.array([G_n[0,2],G_n[1,2], ang]).flatten())


    sk_next = sk + eps*vk

    I_th = 2*IWzz + IBzz*np.cos(sk[0])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(sk[0])**2
    I_th_next = 2*IWzz + IBzz*np.cos(sk_next[0])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(sk_next[0])**2

    # C1 = rw**2/(2*dw**2)*(IBxx-IBzz + mb*b**2)*np.sin(2*sk[0])*(vk[1]-vk[2])**2 
    # C1 += - mb*b*rw*np.sin(sk[0])*vk[0]*(vk[2] + vk[1]) + mb*b*gravity*np.sin(sk[0])
    # C = np.matrix([C1,0,0]).T
    
    # K = rw**2*((mb + 2*mw) - I_th/(2*dw**2))
    # H = rw**2*((mb + 2*mw) + I_th/(2*dw**2))

    # M1 = np.matrix([mb*b**2+IByy, mb*b*rw*np.cos(sk[0]), mb*b*rw*np.cos(sk[0])])
    # M2 = np.matrix([mb*b*rw*np.cos(sk[0]), H + IWyy, K])
    # M3 = np.matrix([mb*b*rw*np.cos(sk[0]), K, H + IWyy])
    # M = np.matrix([M1, M2, M3])


    # M1 = np.matrix([mb*b**2+IByy, mb*b*rw*np.cos(sk_next[0]), mb*b*rw*np.cos(sk_next[0])])
    # M2 = np.matrix([mb*b*rw*np.cos(sk_next[0]), H + IWyy, K])
    # M3 = np.matrix([mb*b*rw*np.cos(sk_next[0]), K, H + IWyy])
    # M_next = np.matrix([M1, M2, M3])
    

    betak = mb*rw*b*np.cos(sk[0])
    betak_next = mb*rw*b*np.cos(sk_next[0])

    xik = I_th/dw**2*rw**2 + IWyy
    xik_next = I_th_next/dw**2*rw**2 + IWyy

    delta = 2*(mb + 2*mw)*rw**2 + IWyy

    D = (xik*(vk[2] - vk[1]) + h*(u[2]-u[1]))/xik_next

    omega = mb*b**2 + IByy

    psik = mb*b*rw*np.cos(sk[0])
    psik_next = mb*b*rw*np.cos(sk_next[0])

    Z = psik*(vk[1]+vk[2]) + omega*vk[0] 
    
    ni = 2*betak*vk[0]/delta + (vk[1]+vk[2]) + h/delta*(u[1]+u[2]) 

    sigmak_next = mb*b*rw*np.sin(sk_next[0])

    C0 = rw**2/(2*dw**2)*(IBxx-IBzz+mb*b**2)*np.sin(2*sk_next[0])*(D**2) + mb*b*gravity*np.sin(sk_next[0])
    C0 += -sigmak_next*ni

    lambd = -2*betak_next/delta*sigmak_next*h
    rho = omega - 2*psik_next*betak_next/delta + h*sigmak_next*ni
    mu = Z + h*u[0] - psik_next*ni + h*C0 

    #print("quadratic eq")
    #print(lambd, rho, mu)

    if lambd != 0:

        if rho**2 < -4*mu*lambd:
            #print("HEY, complex roots!?")
            #print(rho**2 + 4*mu*lambd)
            v_alpha_next = vk[0]
            #v_alpha_next = -rho/(2*lambd)
            #v_alpha_next =  (-rho + np.sqrt(np.abs(rho**2 + 4*mu*lambd)))/(2*lambd)
            
            if stop_on_complex_root:
                print("HEY, complex roots!?")
                print(rho**2 + 4*mu*lambd)
                raise Exception("complex roots encountered")
        else:
            v_alpha_next = (-rho + np.sqrt(rho**2 + 4*mu*lambd))/(2*lambd)
            v_alpha_next_alt = (-rho - np.sqrt(rho**2 + 4*mu*lambd))/(2*lambd)
            #print(v_alpha_next, v_alpha_next_alt)


    elif rho != 0:

        v_alpha_next = mu/rho

    else: 
        raise Exception("Model Error")

    v_phi1_next = - D/2 - betak_next/delta*v_alpha_next + ni/2
    v_phi2_next = + D/2 - betak_next/delta*v_alpha_next + ni/2

    vk_next = [v_alpha_next, v_phi1_next, v_phi2_next]

    #gk_next = [gk_next[0] + gk[0]*np.cos(gk_next[2]) - gk[1]*np.sin(gk_next[2]), gk_next[1] + gk[0]*np.sin(gk_next[2]) + gk[1]*np.cos(gk_next[2]), gk_next[2]]

    qnext = np.asfarray([gk_next, sk_next, vk_next])

    #print(qnext)
    #input()

    return flat(qnext)






def control_policy(t, x, steps=1):
    x_m1 = x[1]
    x = x[0]
    vx = (x[0]-x_m1[0])/eps
    vy = (x[1]-x_m1[1])/eps
    sat = 1
    #c = [0.001*np.sign(x[0])*np.sign(x[1])*max(-sat, min(((vx*2 + vy**2)**0.5)/100, sat))]
    c = [0.01*np.sign(vx*vy)*max(-sat, min(((vx*2 + vy**2)**0.5)/100, sat))]
    if steps == 1:

        return c
    else:
        return c*steps



################### OPTIM




def objective(z, time):
    q = z[:q0.size*time.size].reshape((q0.size, time.size))
    tau = z[q0.size*time.size:].reshape((control_size, time.size))
    h = time[1:] - time[:-1]
    cf = ce = 0
    # cf = 10*np.sum(h*((q[0,1:]-nodes[time.size-1][0])**2 + (q[1,1:]-nodes[time.size-1][1])**2))  # time error/integral cost
    
    # control effort
    ce = 1*(np.sum((tau[:,1:]**2)*h) + 2.5*np.sum((tau[:,0]**2)*h[0]))   

    # nodes error + terminal error
    te = 0
    for n in nodes:
        te += 10*((q[0,n]-nodes[n][0])**2 + (q[1,n]-nodes[n][1])**2)
        # if n != time.size-1:
        #     te += 5*(((q[2,n]- nodes[n][2])%(2*np.pi))**2) 
    # rest conidtion
    te += ((q[3,-1]-nodes[time.size-1][3])**2)
    te += ((q[6,-1]-nodes[time.size-1][6])**2)
    te += ((q[7,-1]-nodes[time.size-1][7])**2)
    te += ((q[8,-1]-nodes[time.size-1][8])**2)

    return cf + te + ce 


def constraint(z, time):
    q = z[:q0.size*time.size].reshape((q0.size, time.size))
    tau = z[q0.size*time.size:].reshape((control_size, time.size))
    res = np.zeros((q0.size, time.size))
    

    # initial values
    res[:, 0] = q[:, 0] - q0


    for j in range(time.size-1):
        h = time[j+1] - time[j]
        # implicite euler scheme
        res[:, j+1] = q[:, j+1] - q[:, j] - h*varIntWIP(time[j+1], q[:, j+1], tau[:,j])

    # h = time[1:] - time[:-1]
    # res[:, 1:] = x[:, 1:] - x[:, :-1] - h * ode_rhs(time, x[:, 1:], v[:-1])

    # NO => you cannot enforce the terminal point, make it convinient via the obj function
    #res[:,-1] = x[:,-1] - np.array([0.5, 0.7])

    return res.flatten()


def opt_constraint(z, time):
    q = z[:q0.size*time.size].reshape((q0.size, time.size))
    g, s, v = np.split(q,3)
    tau = z[(q0.size)*time.size:].reshape((control_size, time.size))
    res = np.zeros(((q0.size), time.size))

    # print(g.shape)
    # print(s.shape)
    # print(v.shape)

    # initial values
    res[:, 0] = q[:, 0] - q0
    zer_vec = np.zeros(time.size-1)
    h = time[1:] - time[:-1]


    Av = np.array(A@v[:,:-1])
    Avm = np.array([[zer_vec,-Av[2,:], Av[0,:]], [Av[2,:], zer_vec, Av[1,:]], [zer_vec,zer_vec,zer_vec]])
    G = np.array([[np.cos(g[2,:-1]), -np.sin(g[2,:-1]), g[0,:-1]], [np.sin(g[2,:-1]), np.cos(g[2,:-1]), g[1,:-1]], [zer_vec,zer_vec,zer_vec+1]])
    # scipy >= 1.81 for batch expm
    G_n = np.einsum("hjk, jlk -> hlk", G, np.transpose(expm(np.transpose(-h*Avm, (2,0,1))),(1,2,0)))
    ang = np.arctan2(G_n[1,0,:],G_n[0,0,:])    # np.arccos(G_n[0,0,:])
    res[:3,1:] = g[:,1:] - np.array([G_n[0,2,:],G_n[1,2,:], ang])

    res[3:6, 1:] = s[:,1:] - s[:,:-1] + v[:,:-1]*h


    I_th = 2*IWzz + IBzz*np.cos(s[0,:-1])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(s[0,:-1])**2
    I_th_next = 2*IWzz + IBzz*np.cos(s[0,1:])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(s[0,1:])**2
    
    #print(I_th.shape, I_th_next.shape)

    betak = mb*rw*b*np.cos(s[0,:-1])
    betak_next = mb*rw*b*np.cos(s[0,1:])

    #print(betak.shape, betak_next.shape)

    xik = I_th/dw**2*rw**2 + IWyy
    xik_next = I_th_next/dw**2*rw**2 + IWyy
    
    #print(xik.shape, xik_next.shape)

    delta = 2*(mb + 2*mw)*rw**2 + IWyy
    
    #print(delta.shape)

    D = (xik*(v[2,:-1] - v[1,:-1]) + h*(tau[1,:-1]-tau[0,:-1]))/xik_next

    #print("D", D.shape)

    omega = mb*b**2 + IByy

    psik = mb*b*rw*np.cos(s[0,:-1])
    psik_next = mb*b*rw*np.cos(s[0,1:])

    #print(psik.shape, psik_next.shape)

    Z = psik*(v[1,:-1]+v[2,:-1]) + omega*v[0,:-1] 

    #print("Z", Z.shape)
    
    ni = 2*betak*v[0,:-1]/delta + (v[1,:-1]+v[2,:-1]) + h/delta*(tau[0,:-1]+tau[1,:-1]) 

    #print("ni", ni.shape)

    sigmak_next = mb*b*rw*np.sin(s[0,1:])

    #print(sigmak_next.shape)

    C0 = rw**2/(2*dw**2)*(IBxx-IBzz+mb*b**2)*np.sin(2*s[0,1:])*(D**2) + mb*b*gravity*np.sin(s[0,1:])
    C0 += -sigmak_next*ni

    #print("C0", C0.shape)



    lambd = -2*betak_next/delta*sigmak_next*h
    rho = omega - 2*psik_next*betak_next/delta + h*sigmak_next*ni
    mu = Z - psik_next*ni + h*C0 

    #print(lambd.shape, rho.shape, mu.shape)

    # v_alpha_next = (-rho + np.sqrt(rho**2 + 4*mu*lambd))/(2*lambd)
    # # for i in range(len(v_alpha_next)):
    # #     if np.isnan(v_alpha_next[i]): 
    # #         v_alpha_next[i] = mu[i]/rho[i]
    # nanvec = np.isnan(v_alpha_next)
    # v_alpha_next_0l = mu[nanvec]/rho[nanvec]
    # v_alpha_next[nanvec] = v_alpha_next_0l

    v_alpha_next = np.zeros_like(rho)
    lambd_nzero = np.argwhere(lambd!=0).flatten()
    lambd_zero = np.invert(lambd_nzero)
    v_alpha_next[lambd_nzero] = (-rho[lambd_nzero] + np.sqrt(rho[lambd_nzero]**2 + 4*mu[lambd_nzero]*lambd[lambd_nzero]))/(2*lambd[lambd_nzero])
    v_alpha_next[lambd_zero] = mu[lambd_zero]/rho[lambd_zero]

    #print(v_alpha_next)

    v_phi1_next = - D/2 - betak_next/delta*v_alpha_next + ni/2
    v_phi2_next = + D/2 - betak_next/delta*v_alpha_next + ni/2

    vk_next = np.array([v_alpha_next, v_phi1_next, v_phi2_next])
    #print(vk_next.shape)

    res[6:9, 1:] = v[:,1:] - vk_next


    #print(res)
    #input()
    return res.flatten()





def implicit_constraint(z, time):
    gravity=9.81
    q = z[:q0.size*time.size].reshape((q0.size, time.size))
    g, s, v = np.split(q,3)
    #zk = z[q0.size*time.size:(q0.size+3)*time.size].reshape((3, time.size))
    #tau = z[(q0.size+3)*time.size:].reshape((control_size, time.size))
    tau = z[(q0.size)*time.size:].reshape((control_size, time.size))
    res = np.zeros(((q0.size), time.size))
    # print(g.shape)
    # print(s.shape)
    # print(v.shape)
    # #print(zk.shape)
    # print(tau.shape)
    
    # initial values
    res[:, 0] = q[:, 0] - q0

    for j in range(time.size-1):
        h = time[j+1] - time[j]
        
        Av = flat(A@v[:,j])
        Avm = np.matrix([[0,-Av[2], Av[0]], [Av[2], 0, Av[1]], [0,0,0]])
        G = np.matrix([[np.cos(g[2,j]), -np.sin(g[2,j]), g[0,j]], [np.sin(g[2,j]), np.cos(g[2,j]), g[1,j]], [0,0,1]])
        G_n = G@expm(-h*Avm)
        ang = np.arctan2(G_n[1,0],G_n[0,0])    # np.arccos(G_n[0,0])
        res[:3,j+1] = g[:,j+1] - np.array([G_n[0,2],G_n[1,2], ang]).flatten()

    #print(res)

    h = time[1:] - time[:-1]
    res[3:6, 1:] = s[:,1:] - s[:,:-1] + v[:,:-1]*h #np.einsum("jk,k -> jk",v[:,:-1],h)
    #print(res)



    for j in range(time.size-1):
        h = time[j+1] - time[j]

        I_th = 2*IWzz + IBzz*np.cos(s[0,j])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(s[0,j])**2
        K = rw**2*((mb + 2*mw) - I_th/(2*dw**2))
        H = rw**2*((mb + 2*mw) + I_th/(2*dw**2))

        M1 = np.matrix([mb*b**2+IByy, mb*b*rw*np.cos(s[0,j]), mb*b*rw*np.cos(s[0,j])])
        M2 = np.matrix([mb*b*rw*np.cos(s[0,j]), H + IWyy, K])
        M3 = np.matrix([mb*b*rw*np.cos(s[0,j]), K, H + IWyy])
        M = np.matrix([flat(M1), flat(M2), flat(M3)])


        I_th_next = 2*IWzz + IBzz*np.cos(s[0,j+1])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(s[0,j+1])**2
        K_next = rw**2*((mb + 2*mw) - I_th_next/(2*dw**2))
        H_next = rw**2*((mb + 2*mw) + I_th_next/(2*dw**2))

        M1 = np.matrix([mb*b**2+IByy, mb*b*rw*np.cos(s[0,j+1]), mb*b*rw*np.cos(s[0,j+1])])
        M2 = np.matrix([mb*b*rw*np.cos(s[0,j+1]), H_next + IWyy, K_next])
        M3 = np.matrix([mb*b*rw*np.cos(s[0,j+1]), K_next, H_next + IWyy])
        M_next = np.matrix([flat(M1), flat(M2), flat(M3)])


        C1 = rw**2/(2*dw**2)*(IBxx-IBzz + mb*b**2)*np.sin(2*s[0,j+1])*(v[1,j+1]-v[2,j+1])**2 
        C1 += - mb*b*rw*np.sin(s[0,j+1])*v[0,j+1]*(v[2,j+1] + v[1,j+1]) + mb*b*gravity*np.sin(s[0,j+1])
        C = np.matrix([C1,0,0]).T

        res[6:9, j+1] = M_next@v[:,j+1] - (h*C).T -M@v[:,j] - h*np.array([0,tau[0,j],tau[1,j]]).T
    
    #print(res)

    # h = time[1:] - time[:-1]
    # res[:, 1:] = x[:, 1:] - x[:, :-1] - h * ode_rhs(time, x[:, 1:], v[:-1])

    return res.flatten()


# vectorized version of implicit_contraint
# 10-60 x speed, even more depending on the constraints
def opt_implicit_constraint(z, time):
    gravity=9.81
    q = z[:q0.size*time.size].reshape((q0.size, time.size))
    g, s, v = np.split(q,3)
    tau = z[(q0.size)*time.size:].reshape((control_size, time.size))
    res = np.zeros(((q0.size), time.size))
    
    # initial values
    res[:, 0] = q[:, 0] - q0

    zer_vec = np.zeros(time.size-1)


    h = time[1:] - time[:-1]

    
    Av = np.array(A@v[:,:-1])
    Avm = np.array([[zer_vec,-Av[2,:], Av[0,:]], [Av[2,:], zer_vec, Av[1,:]], [zer_vec,zer_vec,zer_vec]])
    G = np.array([[np.cos(g[2,:-1]), -np.sin(g[2,:-1]), g[0,:-1]], [np.sin(g[2,:-1]), np.cos(g[2,:-1]), g[1,:-1]], [zer_vec,zer_vec,zer_vec+1]])

    # scipy >= 1.81 for batch expm
    G_n = np.einsum("hjk, jlk -> hlk", G, np.transpose(expm(np.transpose(-h*Avm, (2,0,1))),(1,2,0)))
    ang = np.arctan2(G_n[1,0,:],G_n[0,0,:])    # np.arccos(G_n[0,0,:])
    res[:3,1:] = g[:,1:] - np.array([G_n[0,2,:],G_n[1,2,:], ang])


    res[3:6, 1:] = s[:,1:] - s[:,:-1] + v[:,:-1]*h


    I_th = 2*IWzz + IBzz*np.cos(s[0,:-1])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(s[0,:-1])**2
    K = rw**2*((mb + 2*mw) - I_th/(2*dw**2))
    H = rw**2*((mb + 2*mw) + I_th/(2*dw**2))

    M1 = np.matrix([np.repeat(mb*b**2+IByy,time.size-1), mb*b*rw*np.cos(s[0,:-1]), mb*b*rw*np.cos(s[0,:-1])])
    M2 = np.matrix([mb*b*rw*np.cos(s[0,:-1]), H + IWyy, K])
    M3 = np.matrix([mb*b*rw*np.cos(s[0,:-1]), K, H + IWyy])
    M = np.array([M1, M2, M3])


    I_th_next = 2*IWzz + IBzz*np.cos(s[0,1:])**2 + 2*mw*dw*2 + (IBxx + mb*b**2)*np.sin(s[0,1:])**2
    K_next = rw**2*((mb + 2*mw) - I_th_next/(2*dw**2))
    H_next = rw**2*((mb + 2*mw) + I_th_next/(2*dw**2))

    M1 = np.matrix([np.repeat(mb*b**2+IByy,time.size-1), mb*b*rw*np.cos(s[0,1:]), mb*b*rw*np.cos(s[0,1:])])
    M2 = np.matrix([mb*b*rw*np.cos(s[0,1:]), H_next + IWyy, K_next])
    M3 = np.matrix([mb*b*rw*np.cos(s[0,1:]), K_next, H_next + IWyy])
    M_next = np.array([M1, M2, M3])


    C1 = rw**2/(2*dw**2)*(IBxx-IBzz + mb*b**2)*np.sin(2*s[0,1:])*(v[1,1:]-v[2,1:])**2 
    C1 += - mb*b*rw*np.sin(s[0,1:])*v[0,1:]*(v[2,1:] + v[1,1:]) + mb*b*gravity*np.sin(s[0,1:])
    C = np.array([C1, zer_vec, zer_vec])

    res[6:9, 1:] = np.einsum("ijk, jk-> ik",M_next,v[:,1:]) - C*h - np.einsum("ijk, jk-> ik",M,v[:,:-1])
    res[6:9, 1:] -= np.array([zer_vec,tau[0,:-1],tau[1,:-1]])*h

    return res.flatten()



################### MODEL

g = 9.81
gravity=9.81
e1 = np.asmatrix([1,0], dtype=np.float64).T
e2 = np.asmatrix([0,1], dtype=np.float64).T
m = 1 #mass
L = 1
S = np.asmatrix("[0,-1; 1,0]")

rw = 33.1*10e-3      # wheels radius
dw = 49*10e-3      # semidistance wheels
mw = 0.028      # mass wheel
mb = 0.277      # pend mass
b = 48.67*10e-3       # pend Length
IW = np.matrix([[1,0,0],[0, 7411*10e-6,0],[0,0, 4957*10e-6]])
IB = np.matrix([[543108*10e-6,0,0],[0, 481457*10e-6,0],[0,0, 153951*10e-6]])
IWyy = IW[1,1]
IWzz = IW[2,2]
IBxx = IB[0,0]
IByy = IB[1,1]
IBzz = IB[2,2]
A = np.matrix([[0, -rw, -rw],[0,0,0],[0,rw/dw,-rw/dw]])




os.system("cls")

# time grid
tspan = [0,8]
dt    = 0.1 
time  = np.arange(tspan[0], tspan[1] + dt, dt)

eps = dt
maxiter = 60000 #60 
tol = 5e-3

explicit_model = False

# qk : [gk, sk, vk]
# gk : [x, y, theta]
# sk : [alpha, phi1, phi2]
# vk : [v_alpha, v_phi1, v_phi2]
q0 = flat(np.asarray([1,1,0,0,0,0,0,0,0], dtype=np.float64)) 
qf = flat(np.asarray([1,1,-np.pi/3,0,0,0,0,0,0], dtype=np.float64)) 

nodes = {
    0:  q0,
    20: flat(np.asarray([3,4,0,0,0,0,0,0,0], dtype=np.float64)),
    40: flat(np.asarray([5,1,0,0,0,0,0,0,0], dtype=np.float64)), 
    60: flat(np.asarray([3,-2,-np.pi/6,0,0,0,0,0,0], dtype=np.float64)), 
    time.size-1: qf
}





## INITIALIZATION
init_type = [
    "const",
    "random",
    "steady"
]

fill_init = [
    "nodes_sub",
    "nodes_interpolation"
]

init_type = init_type[1]
fill_init = fill_init[1]  # None


control_size = 2
tau_0 = flat(np.asarray([0,0], dtype=np.float64))
c_ext = lambda q : np.concatenate([q, tau_0],axis=0)
z0_k = c_ext(q0)

# Z0 initialization
if init_type == "const":
    z0 = np.zeros(time.size*(q0.size+control_size)) + 0.1
elif init_type == "random":
    z0 = flat(np.random.rand(time.size*(q0.size+control_size)))
elif init_type == "steady":
    z0 = flat(np.repeat([z0_k], time.size))
else:
    z0 = np.zeros(time.size*(q0.size+control_size)) + 0.1



# Node points subst
if fill_init == "nodes_sub":
    for n in nodes:
        #z0[(q0.size+tau_0.size)*n : (q0.size+tau_0.size)*n+q0.size] = flat(nodes[n])
        z0.reshape((q0.size+tau_0.size, time.size))[:q0.size,n] = nodes[n]

#Linear interpolation
elif fill_init == "nodes_interpolation":
    for i, n in enumerate(nodes):
        if i==0:
            n_pr = n
            continue
        #z0[(q0.size+tau_0.size)*n_pr : (q0.size+tau_0.size)*n] = flat(np.linspace(c_ext(nodes[n_pr]), c_ext(nodes[n]), n-n_pr))
        z0.reshape((q0.size+tau_0.size, time.size))[:,n_pr:n] = np.linspace(c_ext(nodes[n_pr]), c_ext(nodes[n]), n-n_pr).T
        n_pr = n
    z0.reshape((q0.size+tau_0.size, time.size))[:q0.size,-1] = nodes[time.size-1]
    plt.figure()
    plt.plot(z0.reshape((q0.size+tau_0.size, time.size))[0,:], z0.reshape((q0.size+tau_0.size, time.size))[1,:])
    plt.title("intial nodes interpolation")


print(f"Explicit model: {explicit_model}")
print(f"Initialization type: {init_type}")
print(f"Initialization fill: {fill_init}")
print(f"Time size: {time.size}")
print(f"z0 size: {len(z0)}")
print(f"Nodes number: {len(nodes)}")

# variable bounds, here you can set actuator constraints and similar
#bnds = [(None, None) if i < 2*time.size else (-1, 1) for i in range(z0.size)]  # constraints on the control action
bnds = [(None, None) if i < 2*time.size else (None, None) for i in range(z0.size)] # no constraints



if explicit_model:
    # cons = [{
    #     'type': 'eq', 
    #     'fun': lambda z: constraint(z, time),
    #     'jac': lambda z: approx_derivative(lambda zz: constraint(zz, time), z)
    # }]
    cons = [{
        'type': 'eq', 
        'fun': lambda z: opt_constraint(z, time),
        'jac': lambda z: approx_derivative(lambda zz: opt_constraint(zz, time), z)
    }]
else:
    # cons = [{
    #     'type': 'eq', 
    #     'fun': lambda z: implicit_constraint(z, time), 
    #     'jac': lambda z: approx_derivative(lambda zz: implicit_constraint(zz, time), z)
    # }]
    cons = [{
        'type': 'eq', 
        'fun': lambda z: opt_implicit_constraint(z, time), 
        'jac': lambda z: approx_derivative(lambda zz: opt_implicit_constraint(zz, time), z)
    }]


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# call the solver
res = minimize_ipopt(lambda z: objective(z, time), x0=z0, bounds=bnds, 
                     constraints=cons, tol=tol, options = {"maxiter":maxiter,'disp': 5})




print("\n\n\n\n")
print(res)

x = res.info["x"]

print("\n\nResults:\n\n")
print(f"Len result: {len(x)}")
print(f"Final obj: {objective(x,time)}")
q = x[:q0.size*time.size].reshape((q0.size, time.size))
tau = x[q0.size*time.size:].reshape((control_size, time.size))
print("\nFinal position:")
print(q[0,-1], q[1,-1])
print("\nFinal q:")
print(q[:,-1])




vals = {}
for i in range(len(q0)):
    vals[i] = q[i,:]
for i in range(len(tau_0)):
    vals[len(q0) + i] = tau[i,:]
with open("optimal_traj_data.pkl", "wb") as f:
    pk.dump(vals, f)


plt.figure()
plt.plot(np.asarray(tau[0,:]),label="t1")
plt.plot(np.asarray(tau[1,:]),label="t2")
plt.legend()
plt.title("control action")


plt.figure()
for i in range(3):
    plt.plot(q[i,:], label=f"x{i}")
plt.legend()
plt.title("G components")
plt.xlabel("t")
plt.ylabel("x(t)")


plt.figure()
plt.plot(q[0,:], q[1,:])
plt.title("X-Y plot")


plt.figure()
for i in range(3,6):
    plt.plot(q[i,:], label=f"s{i}")
plt.legend()
plt.title("S components")
plt.xlabel("t")
plt.ylabel("s(t)")

plt.figure()
for i in range(6,9):
    plt.plot(q[i,:], label=f"s_d{i-6}")
plt.legend()
plt.title("V components")
plt.xlabel("t")
plt.ylabel("s_d(t)")

#plt.show()
#input()

if __name__ == "__main__":
    
    t0 = 0
    x0 = q0
    gk_gl = x0[:3]

    eps = dt#0.01


    steps = time.size
    sys = dyn.Sys(x0.T, varIntWIP, eps, t0=t0, sys_type="discrete")
    stepped_plot = False
    controlled = True
    expl_energy = False     # se metti True ti plotta l'Energia cinetica, potenziale e totale 
                            # altrimenti solo la totale


    vals = {}
    cvals = []
    for i in range(len(x0)):
        vals[i] = []


    num_updates = 20
    nump_stepped_plots = 20
    for k in range(steps):

        if k % np.floor(steps/num_updates) == 0:
            print(f"Simulation:                                ", end="\r")
            print(f"Simulation: {np.round(k/steps*1000)/10}%", end="\r")

        if controlled:
            #f = -0.3*(sys.x[3]) -0.5*(sys.x[6])
            #u = [[f,f]]
            u = [[tau[0,k],tau[1,k]]]
            sys.control_step(uvec=u)
            cvals.append(u[0])
        else:
            sys.step()
        


        for i in range(len(x0)):
            vals[i].append(sys.x[i]) 


        if stepped_plot and k>0 and k% (steps//nump_stepped_plots) == 0:
            plt.plot(vals[0], label=f"x{0}")
            plt.plot(vals[1], label=f"x{1}")
            plt.legend()
            plt.title("All state components")
            plt.xlabel("t")
            plt.ylabel("x(t)")


            plt.figure()
            plt.plot(vals[0], vals[1])
            plt.ylim(-1,1)
            plt.xlim(-1,1)
            if controlled:
                plt.arrow(sys.x[0], sys.x[1], u[0][0], u[0][1], width = 0.005)

            plt.show()

    print("Simulation Completed!\n\n")


    if not stepped_plot:

        plt.figure()
        for i in range(3):
            plt.plot(vals[i], label=f"x{i}")
        plt.legend()
        plt.title("SIM-All state components")
        plt.xlabel("t")
        plt.ylabel("x(t)")        

        # plt.figure()
        # plt.plot(vals[3], label=f"s{0}")
        # plt.legend()

        plt.figure()
        for i in range(3,6):
            plt.plot(vals[i], label=f"s{i}")
        plt.legend()
        plt.title("SIM-All state components")
        plt.xlabel("t")
        plt.ylabel("s(t)")

        plt.figure()
        for i in range(6,9):
            plt.plot(vals[i], label=f"s_d{i-6}")
        plt.legend()
        plt.title("SIM-All state components")
        plt.xlabel("t")
        plt.ylabel("s_d(t)")


        plt.figure()
        plt.plot(vals[0], vals[1])
        plt.title("SIM-X Y plot")


        if controlled:
            plt.figure()
            plt.plot(cvals)
            plt.title("SIM-Control Action")
            

        #np.save("./data", vals)



        with open("data.pkl", "wb") as f:
            pk.dump(vals, f)

        plt.show()