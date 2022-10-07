import numpy as np
import matplotlib.pyplot as plt

def f(t, x):
    return np.asarray([-0.6*x[1], 0.6*x[0], -0.1*x[2]])


def f2(t, x):
    return np.asarray([-2.6*x[0], -1.2*x[2], 0])

def g(t, x):
    return np.asarray([1, 1, 1])

def glast(t, x):
    return np.asarray([0, 0, 1])

def Lorenz(t, x, sigma=10, rho=28, beta=8/3):
    return np.asarray([sigma*(x[1]-x[0]), rho*x[0] -x[1] - x[0]*x[2], x[0]*x[1]-beta*x[2]])

def strangesin(t, x, T=1):
    return np.asarray([np.cos(t/T)*x])


def pend(t, x, params={"L":1, "g":9.81, "b":0}):
    L = params["L"]
    b = params["b"]
    g = params["g"]
    return np.asarray([x[1], -g/L*np.sin(x[0]) -b*x[1]])



def control_policy(t, x, steps=1):
    if steps == 1:
        c = [-x[2] +x[1]]
        return c
    else:
        return c*steps

def time(sys):
    return sys.clock*sys.eps + sys.t0


# Class to handle parametric models 
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
        self.default_params.update(params) # update with new entries

    def __call__(self, t, x, params={}, fix=False):
        if fix:
            self.update_params(params) 
            parameters = self.default_params
        else:
            parameters = self.default_params.copy()
            parameters.update(params) 

        return self.f(t,x,params=parameters)




class Sys():

    def __init__(self, x_0, f, eps, t0=0, steps=1, g=None, sys_type="continuous", sys_order=1) -> None:
        self.x_0 = x_0          # initial state
        self.f = f              # transition matrix
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
                #self.x = self.f(t, self.x) + self.g(t, self.x)*u
                self.x = self.f(t, self.x, u) 

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


if __name__ == "__main__":

    t0 = 0
    x0 = np.asarray([0, 1, 0])
    x0 = np.asarray([0, -5, 0])
    #x0 = np.asarray([1, 1, 1])
    x0 = np.asarray([0, 10, 10])
    x0 = np.asarray([10, 20, 10])
    t = 2
    eps = 0.01
    T = 1000
    sys = Sys(x0, Lorenz, eps, g=g)
    #sys = Sys(x0, f, eps)
    #sys = Sys(x0, f2, eps, g=glast)

    vals = [x0]
    x = [x0[0]]
    y = [x0[1]]
    z = [x0[2]]
    for i in range(T):
        #print(sys.x)
        #sys.step()
        #sys.control_step(uvec=[i/10])
        sys.control_step(uvec=control_policy(time(sys), sys.x))
        vals.append(sys.x)
        x.append(sys.x[0])
        y.append(sys.x[1])
        z.append(sys.x[2])

    #print(sys.x)
    plt.plot(vals)
    plt.title("All state components")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.figure(2)
    plt.plot(x,y)
    plt.xlabel("x1(t)")
    plt.ylabel("x2(t)")
    plt.figure(3)
    plt.plot(y,z)
    plt.xlabel("x2(t)")
    plt.ylabel("x3(t)")
    plt.figure(4)
    plt.plot(x,z)
    plt.xlabel("x1(t)")
    plt.ylabel("x3(t)")
    plt.figure(5)
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'red')
    plt.title("3D plot")

    plt.show()


