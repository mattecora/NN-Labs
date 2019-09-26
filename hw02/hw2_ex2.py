import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(25)

def label_to_array(label):
    # Transform a label into a binary array
    return np.array([1 if i == label else 0 for i in range(10)]).reshape(10, 1)

def array_to_label(array):
    # Transform a binary array into a label
    return np.argmax(array)

def count_errors(S, n, W, d):
    # Count misclassifications with the given weights
    return sum([1 if array_to_label(W @ S[i]) != d[i] else 0 for i in range(n)])

def multcat_pta(train_samples, train_labels, start_weights, n, eta, eps, max_epochs):
    # Initialize PTA variables
    weights = np.array(start_weights)
    epochs = 0
    
    # Initialize errors array
    errors = [count_errors(train_samples, n, weights, train_labels)]
    print("Epoch {} errors: {} ({:.03%}).".format(epochs, errors[len(errors) - 1], errors[len(errors) - 1] / n))

    # Run the multicategory PTA algorithm
    while errors[epochs] / n > eps and epochs < max_epochs:
        epochs = epochs + 1
        for i in range(n):
            weights = weights + eta * (label_to_array(train_labels[i]) - np.heaviside(weights @ train_samples[i], 1)) @ train_samples[i].transpose()
        errors.append(count_errors(train_samples, n, weights, train_labels))
        print("Epoch {} errors: {} ({:.03%}).".format(epochs, errors[len(errors) - 1], errors[len(errors) - 1] / n))
    
    return weights, epochs, errors

def plot_errors(errors):
    # Create new figure
    plt.figure()

    # Plot data
    plt.plot(errors, marker="x", fillstyle="none")
    
    # Set plot options
    plt.title("Misclassifications per epoch")
    plt.grid()

    # Show plot
    plt.show(block=False)

def idx_images_parse(filename):
    with open(filename, "rb") as f:
        # Check magic number
        if np.fromfile(f, dtype=">u4", count=1)[0] != 2051:
            return False, False, False, False

        # Read number of samples, rows and columns
        n_samples = np.fromfile(f, dtype=">u4", count=1)[0]
        n_rows = np.fromfile(f, dtype=">u4", count=1)[0]
        n_cols = np.fromfile(f, dtype=">u4", count=1)[0]

        # Parse samples
        samples = [np.fromfile(f, dtype=">u1", count=n_rows*n_cols).reshape(n_rows * n_cols, 1) for i in range(n_samples)]

    return n_samples, n_rows, n_cols, samples

def idx_labels_parse(filename):
    with open(filename, "rb") as f:
        # Check magic number
        if np.fromfile(f, dtype=">u4", count=1)[0] != 2049:
            return False, False

        # Read number of labels
        n_labels = np.fromfile(f, dtype=">u4", count=1)[0]

        # Parse labels
        labels = [np.fromfile(f, dtype=">u1", count=1)[0] for i in range(n_labels)]
    
    return n_labels, labels

# Parse the training set
n_train_samples, n_rows, n_cols, train_samples = idx_images_parse("train-images.idx3-ubyte")
_, train_labels = idx_labels_parse("train-labels.idx1-ubyte")
print("Training set loaded.")

# Parse the test set
n_test_samples, _,  _, test_samples = idx_images_parse("t10k-images.idx3-ubyte")
_, test_labels = idx_labels_parse("t10k-labels.idx1-ubyte")
print("Test set loaded.")

# First run: fixed W0, n = 50-1000-60000, eps = 0, eta = 1
W0 = np.random.uniform(-1, 1, size=(10, n_rows*n_cols))
eps = 0
eta = 1
max_epochs = 250

for n in [50, 1000, n_train_samples]:
    print("Running with n = {}, eps = {}, eta = {}.".format(n, eps, eta))

    # Run the multicategory PTA algorithm
    weights, epochs, errors = multcat_pta(train_samples, train_labels, W0, n, eta, eps, max_epochs)
    print("Multicategory PTA finished.")

    # Plot errors per epoch
    plot_errors(errors)

    # Classify test set using computed weights
    test_errors = count_errors(test_samples, n_test_samples, weights, test_labels)
    print("Test set errors: {} ({:.03%}).".format(test_errors, test_errors / n_test_samples))

# Second run: variable W0, n = 60000, eps = 0.125, eta = 1-10-0.1
n = n_train_samples
eps = 0.125
eta = 5
max_epochs = 250

for i in range(3):
    W0 = np.random.uniform(-1, 1, size=(10, n_rows*n_cols))
    print("Running with n = {}, eps = {}, eta = {}.".format(n, eps, eta))

    # Run the multicategory PTA algorithm
    weights, epochs, errors = multcat_pta(train_samples, train_labels, W0, n, eta, eps, max_epochs)
    print("Multicategory PTA finished.")

    # Plot errors per epoch
    plot_errors(errors)

    # Classify test set using computed weights
    test_errors = count_errors(test_samples, n_test_samples, weights, test_labels)
    print("Test set errors: {} ({:.03%}).".format(test_errors, test_errors / n_test_samples))

# Maintain graphs
plt.show()