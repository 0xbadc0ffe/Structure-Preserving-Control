import numpy as np
import matplotlib.pyplot as plt
import torch
import platform, os





if platform.system() == 'Windows':
    CLEAR_STR = "cls" 
else:
    CLEAR_STR = "clear"


os.system(CLEAR_STR)




########################################################
#########     BATCH GENERATION + UTILITIES
########################################################




def rand_L(coef=4, bias=0.3, round=3):  #coeff4
    return np.round(np.random.rand()*coef + bias, round)


def rand_b(coef=2, round=3, prebias=0.5):
    return np.round(max(0, (np.random.rand()-prebias)*coef),     round)

def rand_params():
    res = {}
    for key in parameters_generation:
        f = parameters_generation[key]
        res[key] = f()
    return res


def gen_rollout(sys, x_0, rollout_lenght, t0=0, rand=False, params={}, inc_based=False, device="cpu"):
    if rand:
        params = rand_params()
    #sys.f = lambda t,x : df(t,x,L=L, b=b)
    sys.f.update_params(params)

    sys.restart(x_0.cpu(), t0=t0)

    rollout = [addtime(x_0.cpu(), Sys.time(sys))]
    if inc_based: 
        rollout_lenght+=1
    for i in range(rollout_lenght -1):
        sys.step()
        x_i = sys.x
        rollout.append(addtime(x_i, sys.time()))
    rollout = torch.vstack(rollout).to(device)
    rollout = rollout.unsqueeze(0)
    if inc_based: 
        rollout = torch.concat((rollout[:,1:,:-1] - rollout[:,:-1,:-1], rollout[:,1:,-1:]), dim=2)
        return rollout, x_0, params
    else:
        return rollout, params


def gen_batch(sys, x_0, rollout_lenght, batch_size, basin_r=0, device="cpu", batch_params=None, rand=True):
    
    x0_dev = []
    rollouts = []
    params_list = []
    for j in range(batch_size):
        dev = (torch.rand([len(x_0)], dtype=torch.float, device=device)*2-1)*basin_r
        x0_dev.append(torch.norm(dev.cpu()))
        x_i =  x_0 + dev
        if batch_params is not None:
            rollout, params = gen_rollout(sys, x_i, rollout_lenght, device=device, params=batch_params[j], rand=rand)
        else:
            rollout, params = gen_rollout(sys, x_i, rollout_lenght, device=device, rand=rand)
        rollouts.append(rollout)
        params_list.append(params)
    rollouts = torch.concat(rollouts, dim=0)
    return rollouts, params_list, x0_dev


def addtime(x,t):
    return torch.hstack([x, torch.tensor([t], device=x.device)])






########################################################
#########     Dyn Sys Library
########################################################



def g(t, x):
    return np.asarray([1, 1, 1])

def glast(t, x, n=3):
    vec = np.zeros(n)
    vec[-1] = 1
    return vec


def control_policy(t, x, steps=1):
    if steps == 1:
        c = [-x[2] +x[1]]
        return c
    else:
        return c*steps

def time(sys):
    return sys.clock*sys.eps + sys.t0



# Class that handle parametric models 
class ParametricModel:
    def __init__(self, f, default_params=None) -> None:

        # Transition map
        self.f = f

        # We set the default parameters from the function defaults at first.
        # in this way the user has not to specify all defualt parameters
        # if already defined in f
        self.default_params = f.__defaults__[0]
        if not (type(self.default_params) is dict):
            self.default_params = {}

        if default_params:
            self.default_params.update(default_params)

    def update_params(self, params):
        if not (type(params) is dict):
            params = {"L":params[0,0].item(), "b":params[0,1].item()}
            # p = {}
            # params = params.flatten()
            # for i,k in enumerate(self.default_params):
            #     p[k] = params[i]
            # params = p
        self.default_params.update(params) # update with new entries

    def __call__(self, t, x, params={}, fix=False):
        if fix:
            self.update_params(params) 
            parameters = self.default_params
        else:
            parameters = self.default_params.copy()
            parameters.update(params) 

        return self.f(t,x,params=parameters)




# Class that handle systems integration for differential or discrete transition maps
class Sys():

    def __init__(self, x_0, f, eps, t0=0, steps=1, g=None, sys_type="continuous", sys_order=1) -> None:
        self.x_0 = x_0          # initial state
        self.f = f              # transition function
        if g is None:
            self.g = np.ones(len(x_0))
        else:
            self.g = g
        self.x = x_0            # state
        self.t0 = t0            # initial time
        self.clock = 0          # clock
        self.eps = eps          # epsilon (time step)
        self.steps = steps      # RK steps
        self.sys_type = sys_type  # system type (discrete/continuous)
        self.sys_order = sys_order # order of the system. discrete: sys dyamics uses past sys_order elements. continuous: TODO
        if self.sys_type == "discrete" and self.sys_order > 1:
            self.buff = [x_0]*(sys_order)

    # System step, executes self.steps RK4
    def step(self):
        if self.sys_type == "continuous":
            for i in range(self.steps):
                self.RK4()
                self.clock +=1
        elif self.sys_type == "discrete":
            for i in range(self.steps):
                self.dis_step()
                self.clock +=1
        else:
            raise Exception("Unknwon System Type")

    def control_step(self, uvec):
        if self.sys_type == "continuous":
            for i in range(self.steps):
                self.RK4(control=True, u=uvec[i])
                self.clock +=1
        elif self.sys_type == "discrete":
            for i in range(self.steps):
                self.dis_step(control=True, u=uvec[i])
                self.clock +=1
        else:
            raise Exception("Unknwon System Type")

    def dis_step(self, control=False, u=None):
        t = self.clock*self.eps + self.t0
        if control:
            if self.sys_order > 1:
                self.buff = [self.x] + self.buff[:-1]
                self.x = self.f(t, self.buff) + self.g(t, self.buff)*u
            else:
                self.x = self.f(t, self.x) + self.g(t, self.x)*u

        else:
            if self.sys_order > 1:
                self.buff = [self.x] + self.buff[:-1]
                self.x = self.f(t, self.buff)
            else:
                self.x = self.f(t, self.x)
        


    # Runge-Kutta 4
    def RK4(self, control=False, u=None):
        eps = self.eps
        x = self.x
        f = self.f
        if control:
            g = self.g
            t = self.clock*eps + self.t0
            k1 = eps * f(t, x) + eps*g(t, x)*u
            k2 = eps * f(t + 0.5 * eps, x + 0.5 * k1) + eps * g(t + 0.5 * eps, x + 0.5 * k1)*u
            k3 = eps * f(t + 0.5 * eps, x + 0.5 * k2) + eps * g(t + 0.5 * eps, x + 0.5 * k2)*u
            k4 = eps * f(t + eps, x + k3) + eps * g(t + eps, x + k3)*u
            self.x = x + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

        else:
            t = self.clock*eps + self.t0
            k1 = eps * f(t, x)
            k2 = eps * f(t + 0.5 * eps, x + 0.5 * k1)
            k3 = eps * f(t + 0.5 * eps, x + 0.5 * k2)
            k4 = eps * f(t + eps, x + k3)
            self.x = x + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)


    # System evolves until t_end
    def step_t(self,t_end):
        t = self.clock*self.eps + self.t0
        if t > t_end:
            raise Exception("End time cannot be lower then self.t")
        n = int((t_end -t)/self.eps)
        if self.sys_type == "continuous":
            for i in range(n):
                self.RK4()
                self.clock +=1
        elif self.sys_type == "discrete":
            for i in range(n):
                self.dis_step()
                self.clock +=1
        else:
            raise Exception("Unknwon System Type")


    def restart(self, x0, t0=0):
        self.x0 = x0
        self.t0 = t0
        self.x = x0
        self.clock = 0

    def time(self):
        return self.clock*self.eps + self.t0





########################################################
#########     EXAMPLE: PARAMETRIC HILLS
########################################################



def plot_hill(theta, balls=None):
    plt.figure()
    plt.plot(theta)
    
    if balls is not None:
        balls_pos = balls[:,0]
        x_ind = balls_pos.astype(int)   #np.floor(balls_pos).astype(int)
        x_s = balls_pos - x_ind
        y = x_s*theta[x_ind+1] + (1-x_s)*theta[x_ind]
        plt.scatter(x=balls_pos, y=y, c=np.arange(len(balls_pos))) #cmap="inferno"


def parmetric_hills(t, x, params={"theta":np.asarray([1,0.5,0.2]), "g":9.81, "b":0.05, "dx":1, "size":3, "pacman":False}):

    theta = params["theta"]
    g = params["g"]
    b = params["b"]
    size = params["size"]
    pacman =params["pacman"]
    dy = theta[1:]-theta[0:-1]
    dx = np.zeros_like(dy) + params["dx"]
    alpha = np.arctan2(dy,dx)
    alpha_x = alpha[x[0::2].astype(int)]
    eps = 0.1

    #x[:,1] = (x[:,1] - eps*g*np.sin(alpha_x))*(1-b)
    #x[:,0] = x[:,0] + eps*x[:,1]*np.cos(alpha_x)    #cos because eps*vel is a displacement along the path, not alogn the x axis


    x[1::2] = (x[1::2] - eps*g*np.sin(alpha_x))*(1-b)
    x[0::2] = x[0::2] + eps*x[1::2]*np.cos(alpha_x)    #cos because eps*vel is a displacement along the path, not alogn the x axis


    if pacman:
        x[0::2] = x[0::2] %(size-1)  # pacamn effect
    else:
        x[0::2] = np.clip( x[0::2], 0, size-1.01)

    return x





########################################################
#########     DEFINE MODELS
########################################################



def pend(t, x, params={"L":1, "g":9.81, "b":0}):
    L = params["L"]
    b = params["b"]
    g = params["g"]
    return torch.tensor([x[1], -g/L*np.sin(x[0]) -b*x[1]])


def Lorenz(t, x, params={"sigma":10, "rho":28, "beta":8/3}):
    sigma = params["sigma"]
    rho = params["rho"]
    beta = params["beta"]
    return np.asarray([sigma*(x[1]-x[0]), rho*x[0] -x[1] - x[0]*x[2], x[0]*x[1]-beta*x[2]])



def generic_model(t, x, params={"put here":"default"}):
    param_i = params["..."]
    #...
    # return ordinary diff eqation for continuous systems or
    # the transition map for descrete systems
    return np.array(["..."]) # or torch.tensor([...])





########################################################
#########     MAIN
########################################################



if __name__ == "__main__":


    #########################################
    # EXAMPLE: Case study, pendulum


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    sys_type = "continuous"  #"discrete"                            # system type (Continuous use RK4 as integration step)
    x_0 = torch.tensor([0.5,0], dtype=torch.float, device=device)   # initial condition
    eps = 0.01                                                      # time step
    dym_sys = len(x_0)                                              # sys dimension
    df = ParametricModel(pend)                                      # model
    sys = Sys(x_0, df, eps, sys_type=sys_type)

    # Define what parameters you want to randomize during batch generation and how
    parameters_generation = {
        "L":rand_L, 
        "b":rand_b
    }


    T = 200    # context length

    theta_start = {"L":1, "b":0.1} 
    context_start, _ = gen_rollout(sys, x_0, T, t0=0, rand=False, params=theta_start, inc_based=False, device=device)

    theta_goal = {"L":1.5, "b":0.4}
    context_goal, _ = gen_rollout(sys, x_0, T, t0=0, rand=False, params=theta_goal, inc_based=False, device=device)



    # # Generate a random batch
    # batch_size = 20
    # rollout, params_list, _ = gen_batch(sys, x_0, T, batch_size=batch_size, basin_r=0.4, device=device, rand=True)
    # print(rollout.shape)


    plt.figure()
    plt.plot(context_start[0,:,0].cpu().numpy(), linestyle="dashed", label="C-start")
    plt.plot(context_goal[0,:,0].cpu().numpy(), linestyle="dotted",  label="C-goal")
    plt.title("X0")
    plt.legend()


    plt.figure()
    plt.plot(context_start[0,:,1].cpu().numpy(), linestyle="dashed", label="C-start")
    plt.plot(context_goal[0,:,1].cpu().numpy(), linestyle="dotted", label="C-goal")
    plt.title("X1")
    plt.legend()

    plt.show()



    #########################################
    # # Example of discrete system simulation
    # # NB: state can also not be a vector in this case
    # # I unravled the ball states (seq_length, 2) because it was needed in vector form (seq_length*2)
    # # for something else

    # np.random.seed(42)
    # max_height = 10
    # size = 20
    # num_balls = 100
    # max_height = 2
    # theta = np.random.rand(size)*max_height

    # balls = np.random.rand(num_balls,2)*(size-1)
    # balls[:,1] = 0  # zeroing initial velocity


    # eps = 0.01  # time step
    # df = ParametricModel(parmetric_hills)
    # sys_type="discrete"
    # sys = Sys(balls.reshape(-1), df, eps, sys_type=sys_type)

    # # you can update the model parameters whenever you want, if not the 
    # # simulator will use the default ones
    # params = {"theta":theta, "g":9.81, "b":0.05, "dx":1, "size":size, "pacman":False}
    # sys.f.update_params(params)


    # print("theta:")
    # print(theta)

    # plt.ion()
    # plt.show()

    # vel_track = []
    # vel_track.append(np.mean(balls[:,1]))
    # plot_hill(theta=theta, balls=balls)
    # input("press Enter to start the simulation")
    
    # for j in range(300):

    #     sys.step()
    #     balls[:,0] = sys.x[0::2]
    #     balls[:,1] = sys.x[1::2]
    #     vel_track.append(np.mean(balls[:,1]))
    #     plt.close()
    #     plot_hill(theta=theta, balls=balls)
    #     plt.pause(0.35) #0.2 


    # plt.figure()
    # plt.plot(vel_track)
    # plt.show()



    #########################################

    # # Example of simulation with control action [x_dot = Lorenz(x) + g*u, u:control input ]

    # t0 = 0
    # x0 = np.asarray([10, 20, 10])
    # eps = 0.01
    # T = 1000
    # df = ParametricModel(Lorenz)
    # sys = Sys(x0, df, eps, g=g)


    # vals = [x0]
    # x = [x0[0]]
    # y = [x0[1]]
    # z = [x0[2]]
    # for i in range(T):
    #     sys.control_step(uvec=control_policy(time(sys), sys.x))  #change control_policy to change the control actions
    #     vals.append(sys.x)
    #     x.append(sys.x[0])
    #     y.append(sys.x[1])
    #     z.append(sys.x[2])

    # plt.plot(vals)
    # plt.title("All state components")
    # plt.xlabel("t")
    # plt.ylabel("x(t)")
    # plt.figure(2)
    # plt.plot(x,y)
    # plt.xlabel("x1(t)")
    # plt.ylabel("x2(t)")
    # plt.figure(3)
    # plt.plot(y,z)
    # plt.xlabel("x2(t)")
    # plt.ylabel("x3(t)")
    # plt.figure(4)
    # plt.plot(x,z)
    # plt.xlabel("x1(t)")
    # plt.ylabel("x3(t)")
    # plt.figure(5)
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x, y, z, 'red')
    # plt.title("3D plot")

    # plt.show()


