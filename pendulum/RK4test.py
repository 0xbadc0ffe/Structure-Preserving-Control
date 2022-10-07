import SDG
import torch
import matplotlib.pyplot as plt
import numpy as np


g = 9.81
e1 = np.asmatrix([1,0], dtype=np.float).T
e2 = np.asmatrix([0,1], dtype=np.float).T
m = 1 #mass
L = 1
S = np.asmatrix("[0,-1; 1,0]")



def pend(t, x, params={"L":1, "g":9.81, "b":0}):
    L = params["L"]
    b = params["b"]
    g = params["g"]
    return torch.tensor([x[1], -g/L*np.sin(x[0]) -b*x[1]])


def pol2cart(theta, r=None):
    if r is None:
        return np.stack((L*np.sin(theta), -L*np.cos(theta))).reshape(1,2)
    else:
        return np.stack((r*np.sin(theta), -r*np.cos(theta))).reshape(1,2)

def trasp(arr):
    return np.asmatrix(arr).T

# Energia, se sep=True ti da separatemanete anche i valori di en cinetica e potenziale
def energy(qk, qk_m1, sep=False):
    qk = trasp(qk)
    qk_m1 = trasp(qk_m1)
    T = 0.5*m*np.linalg.norm((qk-qk_m1)/eps)**2
    U = m*g*L*e2.T*(qk+qk_m1)/2
    if sep:
        return np.array([T+U,T,U])
    else:
        return np.ravel(T+U)



device = "cpu"
sys_type = "continuous"  #"discrete"                            # system type (Continuous use RK4 as integration step)
x_0 = torch.tensor([np.pi/2,0], dtype=torch.float, device=device)   # initial condition
eps = 0.001                                                    # time step
dym_sys = len(x_0)                                              # sys dimension
df = SDG.ParametricModel(pend)                                      # model
sys = SDG.Sys(x_0, df, eps, sys_type=sys_type)

# Define what parameters you want to randomize during batch generation and how
parameters_generation = {
    "L":SDG.rand_L, 
    "b":SDG.rand_b
}

simtime = 10
steps = int(np.ceil(simtime/eps))  
time = np.linspace(0, eps*(steps), steps)
T = steps    # context length

theta_start = {"L":1, "b":0.0} 
context_start, _ = SDG.gen_rollout(sys, x_0, T, t0=0, rand=False, params=theta_start, inc_based=False, device=device)


# plt.figure()
# plt.plot(context_start[0,:,0].cpu().numpy())#, linestyle="dashed", label="C-start")
# plt.title("X0")
# plt.legend()


# plt.figure()
# plt.plot(context_start[0,:,1].cpu().numpy(), linestyle="dashed", label="C-start")
# plt.title("X1")
# plt.legend()


T0 = 0                      # initial kinetic energy
E0 = m*g*L*e2.T*pol2cart(x_0[0], r=1).T + T0     # initial total energy
expl_energy = False

if expl_energy:
    E = [np.array([E0, 0, E0]), np.array([E0, 0, E0])]
else:
    E = [E0, E0]

vals = [pol2cart(x_0[0])]
E = []
prev = False
for th in context_start[0,1:,0]:
    v = pol2cart(th)
    vals.append(v)
    if prev is not False:    
        E.append(energy(v,prev,expl_energy))  
    prev = v

vals = np.vstack(vals)
x = vals[:,0].flatten()
y = vals[:,1].flatten()


plt.figure()
plt.plot(time, x, label="x")
plt.plot(time, y, label="y")
plt.legend()
plt.title("x-y components vs time")
plt.ylabel("meters")
plt.xlabel("time")


plt.figure()
plt.plot(x,y)
plt.title("x-y plot")
plt.ylabel("meters")
plt.xlabel("time")

plt.figure()
plt.plot(time[:steps-2], E)
plt.title("total energy")
plt.ylabel("Newton")
plt.xlabel("time")

plt.show()