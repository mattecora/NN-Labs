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

# Define network initial weights
n_hidden = 200
sigma1 = np.sqrt(1 / (n_rows * n_cols + 1))
sigma2 = np.sqrt(1 / (n_hidden + 1))
W1 = np.random.normal(0, sigma1, size=(n_hidden, n_rows * n_cols + 1))
W2 = np.random.normal(0, sigma2, size=(10, n_hidden + 1))

# Train the network
eta = 0.01
eps = 0.01
epoch_limit = 100

net = Network([W1, W2], [phi1, phi2], [der_phi1, der_phi2])

start_time = time()
epochs, errors, training_accuracy, test_accuracy = net.train(train_samples, train_labels, test_samples, test_labels, eta, eps, epoch_limit, "200-10")
print(time() - start_time)