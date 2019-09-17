import matplotlib.pyplot as plt
import numpy as np

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(10)

def u(x):
    # Simple implementation of the step function
    return 1 if x >= 0 else 0

def count_mclass(S, d, w):
    # Count misclassifications with the given weights
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

    print("PTA terminated in {} epochs".format(epochs))
    print("Computed weights: {}".format(w))
    print("Separator equation: m = {}, q = {}".format(-w[1] / w[2], -w[0] / w[2]))
    return w, epochs, mclass

def plot_data_and_sep(S0, S1, w):
    # Create new figure
    plt.figure()

    # Plot data
    plt.plot([x[1] for x in S0], [x[2] for x in S0], marker="s", linestyle="none", fillstyle="none")
    plt.plot([x[1] for x in S1], [x[2] for x in S1], marker="o", linestyle="none", fillstyle="none")
    plt.plot([-1, 1], [-(w[0] + w[1] * x) / w[2] for x in [-1, 1]], linestyle="--")

    # Set plot options
    plt.title("Class separation")
    plt.legend(["$S_0$", "$S_1$", "Boundary"], loc="upper right")
    plt.axis([-1, 1, -1, 1])

    # Show plot
    plt.show(block=False)

def plot_data_and_sep_all(S0, S1, w, eta):
    # Create new figure
    plt.figure()

    # Plot data
    plt.plot([x[1] for x in S0], [x[2] for x in S0], marker="s", linestyle="none", fillstyle="none")
    plt.plot([x[1] for x in S1], [x[2] for x in S1], marker="o", linestyle="none", fillstyle="none")

    for e in eta:
        plt.plot([-1, 1], [-(w[e][0] + w[e][1] * x) / w[e][2] for x in [-1, 1]], linestyle="--")

    # Set plot options
    plt.title("Class separation for different values of $\eta$")
    plt.legend(["$S_0$", "$S_1$", "Boundary (exact)", "Boundary ($\eta=1$)", "Boundary ($\eta=10$)", "Boundary ($\eta=0.1$)"], loc="upper right")
    plt.axis([-1, 1, -1, 1])

    # Show plot    
    plt.show(block=False)

def plot_mclass(mclass, eta):
    # Create new figure
    plt.figure()

    # Plot data
    for e in eta:
        plt.plot(mclass[e], marker="x", fillstyle="none")
    
    # Set plot options
    plt.title("Misclassifications per epoch")
    plt.legend(["$\eta = {}$".format(e) for e in eta], loc="upper right")
    plt.grid()

    # Show plot
    plt.show(block=False)

# Get some random weights
wr = np.array([np.random.uniform(-1/4, 1/4), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
print("Exact weights: {}".format(wr))
print("Separator equation: m = {}, q = {}".format(-wr[1] / wr[2], -wr[0] / wr[2]))

# Build the training set
n = 100
S = [np.array([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) for i in range(n)]
print("Number of samples: {}".format(n))

# Construct S0, S1 and d
d = [1 if wr @ x >= 0 else 0 for x in S]
S0 = [S[i] for i in range(len(S)) if d[i] == 0]
S1 = [S[i] for i in range(len(S)) if d[i] == 1]

print("Samples in S0: {}".format(len(S0)))
print("Samples in S1: {}".format(len(S1)))

# Plot data and separator
plot_data_and_sep(S0, S1, wr)

# Initialize random w0
w0 = np.random.uniform(-1, 1, 3)

# Initialize empty dictionaries
weights = {}
epochs = {}
mclass = {}

# Run PTA with variable eta
for eta in [1, 10, 0.1]:
    weights[eta], epochs[eta], mclass[eta] = pta(eta, w0, S, d)

# Plot separators and misclassification rate
plot_data_and_sep_all(S0, S1, weights, [1, 10, 0.1])
plot_mclass(mclass, [1, 10, 0.1])

# Maintain graphs
plt.show()