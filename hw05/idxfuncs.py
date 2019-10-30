import numpy as np

def label_to_array(label):
    # Transform a label into a binary array
    return np.array([1 if i == label else 0 for i in range(10)]).reshape(10, 1)

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
        samples = [(s - np.mean(s)) / 255 for s in samples]

    return n_samples, n_rows, n_cols, samples

def idx_labels_parse(filename):
    with open(filename, "rb") as f:
        # Check magic number
        if np.fromfile(f, dtype=">u4", count=1)[0] != 2049:
            return False, False

        # Read number of labels
        n_labels = np.fromfile(f, dtype=">u4", count=1)[0]

        # Parse labels
        labels = [label_to_array(np.fromfile(f, dtype=">u1", count=1)[0]) for i in range(n_labels)]
    
    return n_labels, labels