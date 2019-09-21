import math
import numpy as np
import matplotlib.pylab as plt

plt.rc('text', usetex=True)

def step(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, weights, actfun):
        self.weights = np.array(weights)
        self.actfun = actfun
    
    def classify(self, pattern):
        return self.actfun(self.weights @ np.array([1] + pattern))
    
class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
    
    def classify(self, pattern):
        return [n.classify(pattern) for n in self.neurons]

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def classify(self, pattern):
        x = pattern
        for layer in self.layers:
            x = layer.classify(x)
        return x

# First layer (line layer) => 2 inputs, 6 outputs
p11 = Perceptron([1, 1, -1], step)                  # y <= x + 1
p12 = Perceptron([-1, 1, 1], step)                  # y >= -x + 1
p13 = Perceptron([1, -1, 0], step)                  # x <= 1
p14 = Perceptron([-1, -1, 1], step)                 # y >= x + 1
p15 = Perceptron([-2, -1, 0], step)                 # x <= -2
p16 = Perceptron([0, 0, 1], step)                   # y >= 0

l1 = Layer([p11, p12, p13, p14, p15, p16])

# Second layer (NOT layer) => 6 inputs, 6 outputs
p21 = Perceptron([-1/2, 1, 0, 0, 0, 0, 0], step)    # x1
p22 = Perceptron([-1/2, 0, 1, 0, 0, 0, 0], step)    # x2
p23 = Perceptron([-1/2, 0, 0, 1, 0, 0, 0], step)    # x3
p24 = Perceptron([-1/2, 0, 0, 0, 1, 0, 0], step)    # x4
p25 = Perceptron([1/2, 0, 0, 0, 0, -1, 0], step)    # NOT x5
p26 = Perceptron([1/2, 0, 0, 0, 0, 0, -1], step)    # NOT x6

l2 = Layer([p21, p22, p23, p24, p25, p26])

# Third layer (AND layer) => 6 inputs, 2 outputs
p31 = Perceptron([-5/2, 1, 1, 1, 0, 0, 0], step)    # x1 AND x2 AND x3
p32 = Perceptron([-5/2, 0, 0, 0, 1, 1, 1], step)    # x4 AND x5 AND x6

l3 = Layer([p31, p32])

# Fourth layer (OR layer) => 2 inputs, 1 output
p41 = Perceptron([-1/2, 1, 1], step)                # x1 OR x2

l4 = Layer([p41])

# Complete network
net = Network([l1, l2, l3, l4])

# Generate test set
n = 10000
S = [[np.random.uniform(-2.5, 2.5), np.random.uniform(-2.5, 2.5)] for i in range(n)]

# Construct S0, S1 and d
d = [net.classify(x)[0] for x in S]
S0 = [S[i] for i in range(len(S)) if d[i] == 0]
S1 = [S[i] for i in range(len(S)) if d[i] == 1]

# Plot data
plt.plot([x[0] for x in S0], [x[1] for x in S0], marker="s", linestyle="none", fillstyle="none")
plt.plot([x[0] for x in S1], [x[1] for x in S1], marker="o", linestyle="none", fillstyle="none")

# Add title
plt.title("Separation of 10000 samples with the given network")

# Show plot
plt.show()