import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(123)

def rbfmap(x, centers, beta):
    # Map x on its Gaussian RBF representation
    return np.array([np.exp(-beta * np.linalg.norm(x - centers[i]) ** 2) for i in range(len(centers))])

def classify(x, w):
    # Classify a data point from the given weights
    return np.sign(w @ x)

def count_mclass(x, d, w):
    # Count misclassifications with the given params
    return sum([1 if classify(x[i], w) != d[i] else 0 for i in range(len(x))])

def kmeans(k, x):
    print("K-Means started.")

    # Initialize random centroids
    centers = [np.random.uniform(0, 1, size=2) for i in range(k)]

    # Associate each point with its nearest centroid
    classes = [np.argmin([np.linalg.norm(xi - ci) for ci in centers]) for xi in x]

    # Loop until no changes
    i = 0
    unchanged = False

    while not unchanged:
        # Compute the next centroids
        for j in range(k):
            try:
                # Average all points in the Voronoi region
                centers[j] = sum([x[i] for i in range(len(x)) if classes[i] == j]) / len([x[i] for i in range(len(x)) if classes[i] == j])
            except ZeroDivisionError:
                # If the Voronoi region is empty, initialize a random centroid
                centers[j] = np.random.uniform(0, 1, size=2)

        print(f"Iteration {i+1}: Centers = {[np.array2string(x) for x in centers]}")
        i = i + 1

        # Associate each point with its nearest centroid
        new_classes = [np.argmin([np.linalg.norm(xi - ci) for ci in centers]) for xi in x]

        # Check if classification is unchanged
        unchanged = classes == new_classes

        # Update classes
        classes = new_classes

    return centers, classes

def pta(eta, w0, x, d, limit):
    print("PTA started.")

    # Initialize PTA variables
    w = np.array(w0)
    epochs = 0

    # Initialize mclass array
    mclass = []
    mclass.append(count_mclass(x, d, w))

    # Run the PTA algorithm
    while mclass[-1] != 0 and epochs < limit:
        # Increase epochs
        epochs = epochs + 1

        # Loop through samples and update weights
        for i in range(len(x)):
            if d[i] == 1 and classify(x[i], w) == -1:
                w = w + eta * x[i]
            elif d[i] == -1 and classify(x[i], w) == 1:
                w = w - eta * x[i]
        
        # Count misclassifications
        mclass.append(count_mclass(x, d, w))

        # Reduce eta if errors are increasing
        if mclass[-1] > mclass[-2]:
            eta = 0.9 * eta

        print("Epoch {}: Weights = {} - Misclassifications = {} - Eta = {}".format(epochs, w, mclass[-1], eta))

    return w, epochs, mclass

def plot_data(n, xpos, xneg, d, k, pcenters, pclasses, ncenters, nclasses):
    # Create a new figure
    plt.figure(figsize=(8,6))

    # Plot ideal separator
    plt.plot(np.linspace(0, 1, 1000), [0.2 * np.sin(10 * x) + 0.3 for x in np.linspace(0, 1, 1000)], linestyle="--", color="k")
    plt.plot(np.linspace(0.35, 0.65, 100), [np.sqrt(abs(0.15 ** 2 - (x - 0.5) ** 2)) + 0.8 for x in np.linspace(0.35, 0.65, 100)], linestyle="--", color="k")
    plt.plot(np.linspace(0.35, 0.65, 100), [-np.sqrt(abs(0.15 ** 2 - (x - 0.5) ** 2)) + 0.8 for x in np.linspace(0.35, 0.65, 100)], linestyle="--", color="k")

    # Plot positive K-means results
    for j in range(k):
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        if j == 0:
            plt.plot([pcenters[j][0]], [pcenters[j][1]], linestyle="none", marker="o", fillstyle="none", color=color, markersize=8, label="Centers for $C_{1}$")
            plt.plot([xpos[i][0] for i in range(len(xpos)) if pclasses[i] == j], [xpos[i][1] for i in range(len(xpos)) if pclasses[i] == j], linestyle="none", marker="x", fillstyle="none", color=color, label="Samples for $C_{1}$")
        else:
            plt.plot([pcenters[j][0]], [pcenters[j][1]], linestyle="none", marker="o", fillstyle="none", color=color, markersize=8)
            plt.plot([xpos[i][0] for i in range(len(xpos)) if pclasses[i] == j], [xpos[i][1] for i in range(len(xpos)) if pclasses[i] == j], linestyle="none", marker="x", fillstyle="none", color=color)
    
    # Plot negative K-means results
    for j in range(k):
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        if j == 0:
            plt.plot([ncenters[j][0]], [ncenters[j][1]], linestyle="none", marker="s", fillstyle="none", color=color, markersize=8, label="Centers for $C_{-1}$")
            plt.plot([xneg[i][0] for i in range(len(xneg)) if nclasses[i] == j], [xneg[i][1] for i in range(len(xneg)) if nclasses[i] == j], linestyle="none", marker="^", fillstyle="none", color=color, label="Samples for $C_{-1}$")
        else:
            plt.plot([ncenters[j][0]], [ncenters[j][1]], linestyle="none", marker="s", fillstyle="none", color=color, markersize=8)
            plt.plot([xneg[i][0] for i in range(len(xneg)) if nclasses[i] == j], [xneg[i][1] for i in range(len(xneg)) if nclasses[i] == j], linestyle="none", marker="^", fillstyle="none", color=color)

    # Decorate plot
    plt.title(f"Clustering of the training data points ($k = {2*k}$)")
    plt.legend()
    plt.savefig(fname="data.pdf")

def plot_sep(n, x, d, w, k, centers, beta, eval_pts):
    # Create a new figure
    plt.figure(figsize=(8,6))

    # Plot the training set
    plt.plot([x[i][0] for i in range(n) if d[i] == 1], [x[i][1] for i in range(n) if d[i] == 1], linestyle="none", marker="x", fillstyle="none")
    plt.plot([x[i][0] for i in range(n) if d[i] == -1], [x[i][1] for i in range(n) if d[i] == -1], linestyle="none", marker="s", fillstyle="none")

    # Evaluate the separator function on a eval_pts*eval_pts grid
    evals = np.zeros((eval_pts, eval_pts))
    for i in range(eval_pts):
        for j in range(eval_pts):
            evals[j][i] = w @ np.concatenate([[1], rbfmap([i / eval_pts, j / eval_pts], centers, beta)])
        print(f"Separator evaluation: {i+1}/{eval_pts}")

    # Plot the separator
    xx, yy = np.meshgrid(np.linspace(0, 1, eval_pts), np.linspace(0, 1, eval_pts))
    plt.contour(xx, yy, evals, levels=[0], linestyles="dashed")

    # Decorate plot
    plt.legend(["Samples for $C_{1}$", "Samples for $C_{-1}$"])
    plt.title(f"RBF-based perceptron separation ($k = {2*k}, \\beta = {beta}$)")
    plt.savefig(fname="sep.pdf")

# Generate the training set
n = 100
x = [np.random.uniform(0, 1, size=2) for i in range(n)]
d = [1 if x[i][1] < 0.2 * np.sin(10 * x[i][0]) + 0.3 or (x[i][1] - 0.8) ** 2 + (x[i][0] - 0.5) ** 2 < 0.15 ** 2 else -1 for i in range(n)]

# Define the two classes
xpos = [x[i] for i in range(n) if d[i] == 1]
xneg = [x[i] for i in range(n) if d[i] == -1]

# Run the K-means algorithm for the two classes
k = 10
pcenters, pclasses = kmeans(k, xpos)
ncenters, nclasses = kmeans(k, xneg)
centers = pcenters + ncenters

# Plot the training set
plot_data(n, xpos, xneg, d, k, pcenters, pclasses, ncenters, nclasses)

# Map training data on their RBF representation (adding bias component)
beta = 10
xrbf = [rbfmap(x[i], centers, beta) for i in range(len(x))]
xrbf_bias = [np.concatenate([[1], xrbf[i]]) for i in range(len(x))]

# Run PTA
eta = 0.1
w0 = np.random.uniform(-1, 1, size=(2*k + 1))
w, epochs, mclass = pta(eta, w0, xrbf_bias, d, 5000)

# Plot obtained separation
plot_sep(n, x, d, w, k, centers, beta, 250)