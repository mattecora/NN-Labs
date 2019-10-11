import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(2019)

def gradf(xi, yi, w):
    return -2 * np.array([
        sum([yi[i] - w[0] - w[1] * xi[i] for i in range(len(xi))]),
        sum([(yi[i] - w[0] - w[1] * xi[i]) * xi[i] for i in range(len(xi))])
    ]).reshape(2, 1)

def hessf(xi, yi, w):
    return 2 * np.array([
        50,
        sum(xi),
        sum(xi),
        sum([xi[i] ** 2 for i in range(len(xi))])
    ]).reshape(2, 2)

# Create vectors
xi = np.array([i + 1 for i in range(50)])
yi = np.array([x + np.random.uniform(-1, 1) for x in xi])

# Compute linear least squares fit
xm = np.average(xi)
ym = np.average(yi)
m = sum([(xi[i] - xm) * (yi[i] - ym) for i in range(50)]) / sum([(xi[i] - xm)**2 for i in range(50)])
q = ym - m*xm
print("LS fit: m = {}, q = {}".format(m, q))

# Compute weights using gradient descent
eta = 2.25e-5
eps = 1e-6
w = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]).reshape(2, 1)

epochs = 0
while True:
    epochs = epochs + 1
    wnew = w - eta * gradf(xi, yi, w)
    if np.linalg.norm(wnew - w) < eps:
        break

    w = wnew
    if epochs % 1000 == 0:
        print("Epoch {}: w = {}, g = {}".format(epochs, w.transpose(), gradf(xi, yi, w).transpose()))

print("GD fit: m = {}, q = {} (epochs: {})".format(w[1][0], w[0][0], epochs))

# Compute weights after one iteration of Newton's method
w = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]).reshape(2, 1)
w = w - np.linalg.inv(hessf(xi, yi, w)) @ gradf(xi, yi, w)
print("NM fit: m = {}, q = {}".format(w[1][0], w[0][0]))

# Plot the LLS fit
plt.plot(xi, yi, marker="x", linestyle="none")
plt.plot(xi, [m*x + q for x in xi], linestyle="dashed")
plt.title("Linear least squares fit of $(x_i, y_i)$")
plt.show()