import numpy as np
import matplotlib.pyplot as plt

# Load saved weights
err = np.load("err.npy")
W1 = np.load("w1.npy")
W2 = np.load("w2.npy")

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(2019)

# Generate the training set
n_samples = 300
x0 = 0
xn = 1
x = np.random.uniform(x0, xn, size=n_samples)
n = np.random.uniform(-1/10, 1/10, size=n_samples)
d = np.sin(20 * x) + 3 * x + n

# Activation functions
phi1 = lambda v : np.tanh(v)
der_phi1 = lambda v : 1 - np.tanh(v) ** 2
phi2 = lambda v : v
der_phi2 = lambda v : 1

def feedforward(x, W, phi, i):
    v = []
    y = [np.array(x[i]).reshape(1,1)]

    # Compute local fields and outputs
    for j in range(len(W)):
        v.append(W[j] @ np.insert(y[j], 0, 1).reshape(len(y[j]) + 1, 1))
        y.append(phi[j](v[j]).reshape(len(v[j]), 1))

    return v, y

def plot_fit(x, d, W, phi):
    x_eval = np.linspace(0, 1, 100)
    y_eval = [feedforward(x_eval, W, phi, i)[1][-1].item() for i in range(len(x_eval))]

    plt.figure()
    plt.plot(x, d, linestyle="none", marker="o", fillstyle="none")
    plt.plot(x_eval, y_eval)
    plt.title("Fitting obtained using the backpropagation algorithm", fontsize=24)
    plt.legend(["Training set", "Obtained fit"], fontsize=16)

def plot_errors(errors):
    plt.figure()
    plt.plot(errors)
    plt.title("Errors for increasing epochs")
    plt.grid()

# Plot and show results
plot_fit(x, d, [W1, W2], [phi1, phi2])
plot_errors(err)
plt.show()