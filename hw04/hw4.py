import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(2019)

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

def backpropagate(x, d, W, phi, der_phi, i):
    # Compute local fields and outputs
    v, y = feedforward(x, W, phi, i)
    delta = [0] * len(W)
    dEW = [0] * len(W)

    # Compute delta signals
    delta[len(W) - 1] = (d[i] - y[len(W)]) * der_phi[len(W) - 1](v[len(W) - 1])
    for j in reversed(range(len(W) - 1)):
        delta[j] = W[j + 1].transpose()[1:] @ delta[j + 1] * der_phi[j](v[j])

    # Compute the gradient
    for j in range(len(W)):
        dEW[j] = -delta[j] @ np.insert(y[j], 0, 1).reshape(1, len(y[j]) + 1)

    return dEW

def error(x, d, W, phi):
    return sum([(d[i] - feedforward(x, W, phi, i)[1][-1].item()) ** 2 for i in range(len(x))]) / len(x)

def train(x, d, W, phi, der_phi, eta, eps, epoch_limit):
    # Run the gradient descent method
    epochs = 0
    errors = [error(x, d, W, phi)]

    while epochs < epoch_limit and errors[epochs] >= eps:
        # Increment epochs
        epochs = epochs + 1

        # Update weights
        for i in range(len(x)):
            dEW = backpropagate(x, d, W, phi, der_phi, i)
            for j in range(len(dEW)):
                W[j] -= eta * dEW[j]

        # Register the error
        errors.append(error(x, d, W, phi))
        print("Epoch {}: eta = {}, err = {}".format(epochs, eta, errors[epochs]))

        # Decrease eta if necessary
        if errors[epochs] > errors[epochs - 1]:
            eta = 0.9 * eta

    return epochs, errors

def plot_fit(x, d, W, phi):
    x_eval = np.linspace(0, 1, 100)
    y_eval = [feedforward(x_eval, W, phi, i)[1][-1].item() for i in range(len(x_eval))]

    plt.figure()
    plt.plot(x, d, linestyle="none", marker="o", fillstyle="none")
    plt.plot(x_eval, y_eval)
    plt.title("Fitting obtained using the backpropagation algorithm")
    plt.legend(["Training set", "Obtained fit"])

def plot_errors(errors):
    plt.figure()
    plt.plot(errors)
    plt.title("Errors for increasing epochs")
    plt.grid()

# Generate the training set
n_samples = 300
x0 = 0
xn = 1
x = np.random.uniform(x0, xn, size=n_samples)
n = np.random.uniform(-1/10, 1/10, size=n_samples)
d = np.sin(20 * x) + 3 * x + n

# Define network initial weights
n_hidden = 24
sigma1 = np.sqrt(1/n_hidden)
sigma2 = 1
W1 = np.random.normal(0, sigma1, size=(n_hidden, 2))
W2 = np.random.normal(0, sigma2, size=(1, n_hidden + 1))

# Run the training algorithm
eta = 0.1
eps = 0.1
epoch_limit = 5000

epochs, errors = train(x, d, [W1, W2], [phi1, phi2], [der_phi1, der_phi2], eta, eps, epoch_limit)

# Plot the fitting and the errors
plot_fit(x, d, [W1, W2], [phi1, phi2])
plot_errors(errors)

# Save weights and errors to file
np.save("w1.npy", W1)
np.save("w2.npy", W2)
np.save("err.npy", errors)

# Show plots
plt.show()