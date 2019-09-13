import matplotlib.pyplot as plt
import numpy as np

def u(x):
    return 1 if x >= 0 else 0

def count_mclass(S, d, w):
    return sum([1 if u(w @ S[i]) != d[i] else 0 for i in range(len(S))])

def pta(eta, w0, S, d):
    print("Running PTA with eta = {} from w0 = {}".format(eta, w0))

    # Initialize PTA variables
    w = np.array(w0)
    epochs = 0

    # Initialize mclass array
    mclass = []
    mclass.append(count_mclass(S, d, w))

    # Run the PTA algorithm
    while mclass[len(mclass) - 1] != 0:
        epochs = epochs + 1
        for i in range(len(S)):
            w = w + eta * S[i] * (d[i] - u(w @ S[i]))
        mclass.append(count_mclass(S, d, w))
        print("Epoch {}: Weights = {} - Misclassifications = {}".format(epochs, w, mclass[len(mclass) - 1]))

    return w, epochs, mclass

def plot_data_and_sep(S0, S1, w, eta=None):
    plt.figure()
    plt.plot([x[1] for x in S0], [x[2] for x in S0], color="r", marker="s", linestyle="none", fillstyle="none")
    plt.plot([x[1] for x in S1], [x[2] for x in S1], color="b", marker="o", linestyle="none", fillstyle="none")
    plt.plot([-1, 1], [-(w[0] + w[1] * x) / w[2] for x in [-1, 1]], color="g", linestyle="--")
    plt.axis([-1, 1, -1, 1])
    plt.title("Separation (eta: {})\nWeights: {}".format(eta, w))
    plt.legend(["S0", "S1", "Boundary"])
    plt.show(block=False)

def plot_mclass(mclass, eta):
    plt.figure()
    plt.plot(mclass, marker="o", linestyle="--", fillstyle="none")
    plt.axis([0, len(mclass) - 1, 0, max(mclass)])
    plt.title("Misclassifications per epoch (eta: {})".format(eta))
    plt.grid()
    plt.show(block=False)

# Get some random weights
wr = np.array([np.random.uniform(-1/4, 1/4), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
print("Exact weights: {}".format(wr))

# Build the training set
n_samples = 1000
S = [np.array([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) for i in range(n_samples)]
print("Number of samples: {}".format(n_samples))

# Construct S0, S1 and d
S0 = [x for x in S if wr @ x >= 0]
S1 = [x for x in S if wr @ x < 0]
d = [1 if wr @ x >= 0 else 0 for x in S]

print("Samples in S0: {}".format(len(S0)))
print("Samples in S1: {}".format(len(S1)))

# Plot data and separator
plot_data_and_sep(S0, S1, wr)

# Initialize random w0
w0 = np.random.uniform(-1, 1, 3)

# Run PTA with variable eta
for eta in [1, 10, 0.1]:
    print("================")
    w, epochs, mclass = pta(eta, w0, S, d)

    # Plot separator and misclassification rate
    plot_data_and_sep(S0, S1, w, eta)
    plot_mclass(mclass, eta)

# Maintain graphs
plt.show()