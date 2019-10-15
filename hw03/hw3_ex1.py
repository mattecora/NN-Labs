import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(2019)

def f(x, y):
    return - np.log(1-x-y) - np.log(x) - np.log(y)

def gradf(x, y):
    return np.array([
        1/(1-x-y) - 1/x,
        1/(1-x-y) - 1/y
    ]).reshape(2, 1)

def hessf(x, y):
    return np.array([
        1/((1-x-y)**2) + 1/(x**2),
        1/((1-x-y)**2),
        1/((1-x-y)**2),
        1/((1-x-y)**2) + 1/(y**2)
    ]).reshape(2, 2)

def gdesc(xy0, eta, eps):
    # Initialize parameters
    xy = np.array(xy0).reshape(2, 1)
    xylist = [np.copy(xy)]
    fxy = [f(xy[0], xy[1])]
    
    # Run the algorithm until the gradient is nearly zero
    while True:
        xynew = xy - eta*gradf(xy[0], xy[1])
        if np.linalg.norm(xynew - xy) < eps:
            break

        xy = xynew
        xylist.append(np.copy(xy))
        fxy.append(f(xy[0], xy[1]))
    
    return xy, xylist, fxy

def newton(xy0, eta, eps):
    # Initialize parameters
    xy = np.array(xy0).reshape(2, 1)
    xylist = [np.copy(xy)]
    fxy = [f(xy[0], xy[1])]

    # Run the algorithm until the gradient is nearly zero
    while True:
        xynew = xy - eta * np.linalg.inv(hessf(xy[0], xy[1])) @ gradf(xy[0], xy[1])
        if np.linalg.norm(xynew - xy) < eps:
            break

        xy = xynew
        xylist.append(np.copy(xy))
        fxy.append(f(xy[0], xy[1]))
    
    return xy, xylist, fxy

def plot_traj(xylist, xymin):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a triangular domain and plot the function
    xv = []
    yv = []
    delta = 0.025
    step = 0.01

    x = delta
    while x < 1 - delta:
        y = delta
        while y < 1 - delta - x:
            xv.append(float(x))
            yv.append(float(y))
            y = y + step
        x = x + step

    xv = np.array(xv)
    yv = np.array(yv)

    ax.plot_trisurf(xv, yv, f(xv,yv), color=(1, 1, 1, 0.25))
    ax.plot([x[0][0] for x in xylist], [x[1][0] for x in xylist], [f(x[0][0], x[1][0]) for x in xylist], marker="x", linestyle="dotted")
    ax.plot([xymin[0]], [xymin[1]], [f(xymin[0], xymin[1])], marker="o", fillstyle="none")

    ax.set_title("Convergence of $(x,y)$ to the minimum of $f$")
    fig.tight_layout()

def plot_fxy(fxy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fxy)
    ax.set_title("Value of $f(x,y)$ for increasing epochs")
    ax.grid()

# Set algorithm parameters
eta = 0.01
eps = 1e-6
xymin = np.array([1/3, 1/3])

# Select random initial point
x0 = np.random.uniform(0, 1)
y0 = np.random.uniform(0, 1-x0)
print("Initial point: ({}, {})".format(x0, y0))
print("GD update term in (x0, y0):\n{}".format(gradf(x0, y0)))
print("NM update term in (x0, y0):\n{}".format(np.linalg.inv(hessf(x0, y0)) @ gradf(x0, y0)))

# Run gradient descent
xy_gd, xylist_gd, fxy_gd = gdesc([x0, y0], eta, eps)
print("Gradient descent iterations: {}".format(len(xylist_gd)))
print("Gradient descent convergence point: {}".format(xy_gd.transpose()))

# Plot results
plot_traj(xylist_gd, xymin)
plot_fxy(fxy_gd)

# Run Newton's method
eta = 1
eps = 1e-6

xy_n, xylist_n, fxy_n = newton([x0, y0], eta, eps)
print("Newton's method iterations: {}".format(len(xylist_n)))
print("Netwon's method convergence point: {}".format(xy_n.transpose()))

# Plot results
plot_traj(xylist_n, xymin)
plot_fxy(fxy_n)

# Show plots
plt.show()