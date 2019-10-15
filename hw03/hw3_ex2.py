import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(2019)

def gradf(xi, yi, w, start, end):
    return -2 * np.array([
        sum([yi[i] - w[0] - w[1] * xi[i] for i in range(start, end)]),
        sum([(yi[i] - w[0] - w[1] * xi[i]) * xi[i] for i in range(start, end)])
    ]).reshape(2, 1)

def hessf(xi, yi, w, start, end):
    return 2 * np.array([
        50,
        sum(xi[start:end]),
        sum(xi[start:end]),
        sum([xi[i] ** 2 for i in range(start, end)])
    ]).reshape(2, 2)

def batch_gdesc(w0, eta, eps, bsize):
    # Initialize parameters
    w_gd = np.array(w0)
    epochs = 0

    while True:
        # Increment epochs and save old weights
        epochs = epochs + 1
        w_old = np.array(w_gd)

        # Update summing according to the batch size
        for i in range(50 // bsize):
            w_gd = w_gd - eta * gradf(xi, yi, w_gd, i*bsize, (i+1)*bsize)
        
        # Return if the norm of the update is small enough
        if np.linalg.norm(w_gd - w_old) < eps:
            return w_gd, epochs
        
        # Print message every 1000 epochs
        if epochs % 1000 == 0:
            print("Epoch {}: w = {}".format(epochs, w_gd.transpose()))

# Create vectors
xi = [i + 1 for i in range(50)]
yi = [x + np.random.uniform(-1, 1) for x in xi]

# Compute linear least squares fit
X = np.array([1] * 50 + xi).reshape(2, 50)
Y = np.array(yi).reshape(1, 50)
w_ls = (Y @ np.linalg.pinv(X)).transpose()
print("LS fit: m = {}, q = {}".format(w_ls[1][0], w_ls[0][0]))

# Compute weights using gradient descent
w0 = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]).reshape(2, 1)
eta = 2.25e-5
eps = 1e-6
bsize = 1

w_gd, epochs = batch_gdesc(w0, eta, eps, bsize)
print("GD fit: m = {}, q = {} (epochs: {})".format(w_gd[1][0], w_gd[0][0], epochs))

# Compute weights after one iteration of Newton's method
w_nm = w0 - np.linalg.inv(hessf(xi, yi, w0, 0, 50)) @ gradf(xi, yi, w0, 0, 50)
print("NM fit: m = {}, q = {}".format(w_nm[1][0], w_nm[0][0]))

# Plot the LLS fit
plt.plot(xi, yi, marker="x", linestyle="none")
plt.plot(xi, [w_ls[1][0]*x + w_ls[0][0] for x in xi], linestyle="dashed")
plt.title("Linear least squares fit of $(x_i, y_i)$")
plt.show()