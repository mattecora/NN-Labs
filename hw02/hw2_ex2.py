import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

def count_errors(S, n, W, d):
    # Count misclassifications with the given weights
    return sum([np.argmax(W @ S[i]) != d[i] for i in range(n)])

def label_to_array(label):
    # Transform a label into a binary array
    a = np.zeros(10).reshape(10, 1)
    a[label] = 1
    return a

def multcat_pta(train_samples, train_labels, start_weights, n, eta, eps, max_epochs):
    # Initialize W randomly
    weights = np.array(start_weights)

    # Initialize parameters
    epochs = 0
    errors = []
    errors.append(count_errors(train_samples, n, weights, train_labels))

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
        # Discard magic number
        np.fromfile(f, dtype=">i", count=1)

        # Read number of samples, rows and columns
        n_samples = np.fromfile(f, dtype=">u4", count=1)[0]
        n_rows = np.fromfile(f, dtype=">u4", count=1)[0]
        n_cols = np.fromfile(f, dtype=">u4", count=1)[0]

        # Parse samples
        samples = [np.fromfile(f, dtype=">u1", count=n_rows*n_cols).reshape(n_rows * n_cols, 1) for i in range(n_samples)]

    return n_samples, n_rows, n_cols, samples

def idx_labels_parse(filename):
    with open(filename, "rb") as f:
        # Discard magic number
        np.fromfile(f, dtype=">i", count=1)

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
W0 = np.random.uniform(size=(10, n_rows*n_cols))
nv = [50, 1000, n_train_samples]
epsv = [0, 0, 0]
etav = [1, 1, 1]
max_epochsv = [250, 250, 250]

for i in range(len(nv)):
    n, eps, eta, max_epochs = nv[i], epsv[i], etav[i], max_epochsv[i]
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
nv = [n_train_samples, n_train_samples, n_train_samples]
epsv = [0.125, 0.125, 0.125]
etav = [1, 10, 0.1]
max_epochsv = [250, 250, 250]

for i in range(len(nv)):
    W0 = np.random.uniform(size=(10, n_rows*n_cols))
    n, eps, eta, max_epochs = nv[i], epsv[i], etav[i], max_epochsv[i]
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