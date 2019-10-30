from sys import argv
from time import time
import numpy as np

from network import Network
from idxfuncs import idx_images_parse, idx_labels_parse

# Set the random seed for reproducibility
np.random.seed(2019)

# Activation functions
phi1 = lambda v : np.tanh(v)
der_phi1 = lambda v : 1 - np.tanh(v) ** 2
phi2 = lambda v : 1 / (1 + np.exp(-v))
der_phi2 = lambda v : np.exp(-v) / (1 + np.exp(-v)) ** 2

# Parse the training set
n_train_samples, n_rows, n_cols, train_samples = idx_images_parse("../hw02/train-images.idx3-ubyte")
_, train_labels = idx_labels_parse("../hw02/train-labels.idx1-ubyte")
print("Training set loaded.")

# Parse the test set
n_test_samples, _,  _, test_samples = idx_images_parse("../hw02/t10k-images.idx3-ubyte")
_, test_labels = idx_labels_parse("../hw02/t10k-labels.idx1-ubyte")
print("Test set loaded.")

# Parse number of neurons from command line
neurons = [n_rows * n_cols] + [int(argv[i]) for i in range(1, len(argv))] + [10]

# Define standard deviations for each layer
sigma = [np.sqrt(1 / (neurons[i] + 1)) for i in range(len(neurons) - 1)]

# Initialize weights for each layer
W = [np.random.normal(0, sigma[i], size=(neurons[i + 1], neurons[i] + 1)) for i in range(len(sigma))]

# Define activation functions for each layer
phi = [phi1] * (len(W) - 1) + [phi2]
der_phi = [der_phi1] * (len(W) - 1) + [der_phi2]

# Create the network
net = Network(W, phi, der_phi)

# Train the network
eta = 0.01
eps = 0.01
epoch_limit = 100

start_time = time()
epochs, errors, training_accuracy, test_accuracy = net.train(train_samples, train_labels, test_samples, test_labels, eta, eps, epoch_limit, " ".join(f"{n}" for n in neurons))
print(f"Elapsed time: {time() - start_time}")