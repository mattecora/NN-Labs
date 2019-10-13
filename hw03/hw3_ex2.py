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
xi = [i + 1 for i in range(50)]
yi = [x + np.random.uniform(-1, 1) for x in xi]

# Compute linear least squares fit
X = np.array([1] * 50 + xi).reshape(2, 50)
Y = np.array(yi).reshape(1, 50)
w_ls = (Y @ np.linalg.pinv(X)).transpose()
print("LS fit: m = {}, q = {}".format(w_ls[1][0], w_ls[0][0]))

# Compute weights using gradient descent
eta = 2.25e-5
eps = 1e-6
w_gd = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]).reshape(2, 1)

epochs = 0
while True:
    epochs = epochs + 1
    wnew = w_gd - eta * gradf(xi, yi, w_gd)
    if np.linalg.norm(wnew - w_gd) < eps:
        break

    w_gd = wnew
    if epochs % 1000 == 0:
        print("Epoch {}: w = {}, g = {}".format(epochs, w_gd.transpose(), gradf(xi, yi, w_gd).transpose()))

print("GD fit: m = {}, q = {} (epochs: {})".format(w_gd[1][0], w_gd[0][0], epochs))

# Compute weights after one iteration of Newton's method
w_nm = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]).reshape(2, 1)
w_nm = w_nm - np.linalg.inv(hessf(xi, yi, w_nm)) @ gradf(xi, yi, w_nm)
print("NM fit: m = {}, q = {}".format(w_nm[1][0], w_nm[0][0]))

# Plot the LLS fit
plt.plot(xi, yi, marker="x", linestyle="none")
plt.plot(xi, [w_ls[1][0]*x + w_ls[0][0] for x in xi], linestyle="dashed")
plt.title("Linear least squares fit of $(x_i, y_i)$")
plt.show()