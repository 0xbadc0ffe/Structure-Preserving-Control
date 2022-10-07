import numpy as np
import matplotlib.pyplot as plt





# Step della dinamica trvoata con il variational integrator
def step(qk, qk_m1):
    inv = np.hstack((S@qk, qk))
    val = qk.T@S@qk_m1 - h**2*g/L*e2.T@S@qk
    #print(val)
    res = np.hstack((val, np.asmatrix([1])))
    return res@inv**(-1)

# Step della dinamica trovta discretizzando l'equazione differenziale in coordinate polari
def euler_angular_step_fwd(thetak, thetak_m1):

    #thetak = np.arctan2(qk[1],qk[0])
    #thetak_m1 = np.arctan2(qk_m1[1],qk_m1[0])

    # discretizing d^2theta/dt^2 + g/L*theta=0
    # thetak_next = 2*thetak -thetak_m1 - h^2*g/L*thetak_m1

    thetak_next = 2*thetak -thetak_m1 - h**2*g/L*thetak_m1   
    thetak_next = my_squeeze(thetak_next)
    return thetak_next


#TODO
# Step della dinamica trovta discretizzando l'equazione differenziale in coordinate polari
def euler_angular_step_bkw(thetak, thetak_m1):

    #thetak = np.arctan2(qk[1],qk[0])
    #thetak_m1 = np.arctan2(qk_m1[1],qk_m1[0])

    # discretizing d^2theta/dt^2 + g/L*theta=0
    # thetak_next = 2*thetak -thetak_m1 - h^2*g/L*thetak_m1

    thetak_next = 2*thetak -thetak_m1 - h**2*g/L*thetak_m1   
    thetak_next = my_squeeze(thetak_next)
    return thetak_next


# Step della dinamica trovta discretizzando l'equazione differenziale in coordinate polari
def euler_angular_step(thetak, thetak_m1, linearized=False):

    #thetak = np.arctan2(qk[1],qk[0])
    #thetak_m1 = np.arctan2(qk_m1[1],qk_m1[0])

    # discretizing d^2theta/dt^2 + g/L*theta=0
    # thetak_next = 2*thetak -thetak_m1 - h^2*g/L*thetak

    if linearized:
        thetak_next = 2*thetak -thetak_m1 - h**2*g/L*thetak
    else:
        thetak_next = 2*thetak -thetak_m1 - h**2*g/L*np.sin(thetak)    # + h**2*g/L*thetak  to get something cool
    thetak_next = my_squeeze(thetak_next)
    return thetak_next


# Step della dinamica trovata discretizzando la dinamica ottenuta dall'eq lagrangiana cartesiana
def euler_step(qk, qk_m1):
    qk_next = 2*qk - qk_m1 - my_squeeze(qk.T@qk - 2*qk.T@qk_m1 + qk_m1.T@qk_m1)[0]*qk_m1   
    #qk_next = 2*qk - qk_m1 - h**2*my_squeeze(np.linalg.norm((qk-qk_m1)/h)**2)[0]*qk_m1
    qk_next -= g*h**2/L*(np.eye(2) - qk_m1@qk_m1.T)@e2
    return qk_next

# Energia, se sep=True ti da separatemanete anche i valori di en cinetica e potenziale
def energy(qk,qk_m1, sep=False):
    T = 0.5*m*np.linalg.norm((qk-qk_m1)/h)**2
    U = m*g*L*e2.T*(qk+qk_m1)/2
    if sep:
        return np.array([T+U,T,U])
    else:
        return np.ravel(T+U)


def my_squeeze(mat):
    return np.array(mat).ravel()

# Da polare (angolo zero all'eq stabile) a cartesiano
def pol2cart(theta, r=None):
    if r is None:
        return np.stack((L*np.sin(theta), -L*np.cos(theta))).reshape(1,2)
    else:
        return np.stack((r*np.sin(theta), -r*np.cos(theta))).reshape(1,2)



e1 = np.asmatrix([1,0]).T
e2 = np.asmatrix([0,1]).T


## CONDIZIONI
S = np.asmatrix("[0,-1; 1,0]")
g = 9.81
m = 1
h = 0.1 #0.01   # 0.001 for euler cart
simtime = 100
steps = int(np.ceil(simtime/h))   #8000
#L = 1

#q0 = np.asmatrix([1.,0.]).T 
#q0 = np.asmatrix([0.5, 0.75**0.5]).T 
q0 = pol2cart(np.pi/2, r=1).T

L = np.linalg.norm(q0)

expl_energy = False     # se metti True ti plotta l'Energia cinetica, potenziale e totale 
                        # altrimenti solo la totale


# QUI CAMBI IL MODELLO DELLA DINAMICA
mode_list = ["variational integrator", "polar", "polar fwd", "euler cart"]
mode = mode_list[0]



qk = q0
qk_m1 = q0
thetak = (np.arctan2(qk[1],qk[0])+np.pi/2)#%2*np.pi
thetak_m1 = thetak


print(f"\nSimulated time: {h*steps} sec   [{steps} steps| h: {h}]")
print("\nL: ", L)
print("\nS: \n", S)
print("\nq0: \n", q0)
print(f"\nModel: {mode}")
print("\nTheta0: ", thetak)
print("\nEn iniziale: ", energy(q0,q0), "\n\n")

i=0
state_plot = q0
theta_plot = [thetak]
T0 = 0                      # initial kinetic energy
E0 = m*g*L*e2.T*q0 + T0     # initial total energy

if expl_energy:
    E = [np.array([E0, 0, E0]), np.array([E0, 0, E0])]
else:
    E = [E0, E0]

E2 = []

max_elong_steps = np.floor(2*np.pi*(L/g)**0.5*1/(2*h)) # pendulum period/(2*h)
max_elong_def_val = -100
max_elong = q0[1]
max_elong_buff = [max_elong]
print(f"Max elongation steps: {max_elong_steps}\n\n")

num_updates=20 
while i<steps:
    if i % np.floor(steps/num_updates) == 0:
        print(f"Simulation:                                ", end="\r")
        print(f"Simulation: {np.round(i/steps*1000)/10}%", end="\r")
    if mode == "variational integrator":
        tmp = step(qk, qk_m1).T
        qk_m1 = qk
        qk = tmp
    elif mode == "polar":
        tmp = euler_angular_step(thetak, thetak_m1).T
        thetak_m1 = thetak
        thetak = tmp
        qk = pol2cart(thetak).T
        qk_m1 = pol2cart(thetak_m1).T
        theta_plot.append(thetak)
    elif mode == "polar fwd":
        tmp = euler_angular_step(thetak, thetak_m1).T
        thetak_m1 = thetak
        thetak = tmp
        qk = pol2cart(thetak).T
        qk_m1 = pol2cart(thetak_m1).T
        theta_plot.append(thetak)
    elif mode == "euler cart":
        tmp = euler_step(qk, qk_m1)
        qk_m1 = qk
        qk = tmp
    #print(qk)
    if i>=1:
        E.append(energy(qk,qk_m1,expl_energy))  
        E2.append([0.5*m*((thetak-thetak_m1)/h)**2*L**2, 0.5*m*L**2*np.linalg.norm((qk-qk_m1)/h)**2])
        #E.append(0.5*m*((thetak-thetak_m1)/h)**2*L**2 + (-np.cos(thetak)*L)*g*m)
    state_plot = np.hstack((state_plot,qk))
    if i % max_elong_steps == 0:
        max_elong = max_elong_def_val
    else:
        if qk[1]>max_elong:
            max_elong = qk[1]
    max_elong_buff.append(max_elong)
    i +=1

print("Simulation Completed!\n\n")

time = np.linspace(0,h*steps,steps+1)


x1 = np.array(state_plot[0,:]).flatten()
x2 = np.array(state_plot[1,:]).flatten()
plt.plot(time, x1, label="x")
plt.plot(time, x2, label="y")
plt.legend()
plt.title("x-y components vs time")
plt.ylabel("meters")
plt.xlabel("time")

plt.figure()
plt.plot(x1, x2)
plt.title("x-y plot")
plt.ylabel("meters")
plt.xlabel("time")

if mode == "polar" or mode == "polar fwd":
    plt.figure()
    plt.plot(time, theta_plot)

plt.figure()
#plt.plot(time, E)
plt.plot(time, E)
plt.title("total energy")
plt.ylabel("Newton")
plt.xlabel("time")

plt.figure()
plt.plot(E2)

plt.figure()
plt.plot(max_elong_buff)
# print(max_elong_buff[:20])
# print(max_elong_buff[-20:])

plt.show()
#plt.savefig("lol.png")