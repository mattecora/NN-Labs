import numpy as np
import matplotlib.pyplot as plt
import qpsolvers as qp

# Use TeX for text rendering
plt.rc('text', usetex=True)

# Set the random seed for reproducibility
np.random.seed(123)

def kern_poly(d):
    return lambda x, y : (1 + x.transpose() @ y) ** d

def kern_gaus(c):
    return lambda x, y : np.exp(-c * np.linalg.norm(x - y) ** 2)

def solve_svm(n, x, d, kernel, tol):
    # Define matrices for quadprog
    P = np.array([d[i] * d[j] * kernel(x[i], x[j]) for i in range(n) for j in range(n)]).reshape(n, n) + 1e-9 * np.eye(n)
    q = -np.ones(n)
    G = -np.eye(n)
    h = np.zeros(n)
    A = np.array(d)
    b = np.zeros(1)

    # Run the solver
    a = qp.solve_qp(P, q, G, h, A, b)

    # Remove all small alphas
    a[a < tol] = 0
    
    # Find the support vectors
    isv = np.nonzero(a)[0]
    asv = [a[i] for i in isv]
    xsv = [x[i] for i in isv]
    dsv = [d[i] for i in isv]

    # Compute the separator
    theta = dsv[0] - sum([asv[i] * dsv[i] * kernel(xsv[i], xsv[0]) for i in range(len(isv))])
    sep = lambda z : sum([asv[i] * dsv[i] * kernel(xsv[i], z) for i in range(len(isv))]) + theta

    # Return separator and support vectors
    return sep, xsv, dsv

def plot_data(n, x, d):
    # Create a new figure
    plt.figure()

    # Plot the training set
    plt.plot([x[i][0] for i in range(n) if d[i] == 1], [x[i][1] for i in range(n) if d[i] == 1], linestyle="none", marker="x", fillstyle="none")
    plt.plot([x[i][0] for i in range(n) if d[i] == -1], [x[i][1] for i in range(n) if d[i] == -1], linestyle="none", marker="s", fillstyle="none")

    # Plot ideal separator
    plt.plot(np.linspace(0, 1, 1000), [0.2 * np.sin(10 * x) + 0.3 for x in np.linspace(0, 1, 1000)], linestyle="--", color="k")
    plt.plot(np.linspace(0.35, 0.65, 100), [np.sqrt(abs(0.15 ** 2 - (x - 0.5) ** 2)) + 0.8 for x in np.linspace(0.35, 0.65, 100)], linestyle="--", color="k")
    plt.plot(np.linspace(0.35, 0.65, 100), [-np.sqrt(abs(0.15 ** 2 - (x - 0.5) ** 2)) + 0.8 for x in np.linspace(0.35, 0.65, 100)], linestyle="--", color="k")

    # Decorate plot
    plt.legend(["$C_{1}$", "$C_{-1}$", "Ideal separator"])
    plt.title("Training data points")
    plt.savefig(fname="data.pdf")

def plot_sep(n, x, d, xsv, sep, eval_pts):
    # Create a new figure
    plt.figure()

    # Plot the training set
    plt.plot([x[i][0] for i in range(n) if d[i] == 1], [x[i][1] for i in range(n) if d[i] == 1], linestyle="none", marker="x", fillstyle="none")
    plt.plot([x[i][0] for i in range(n) if d[i] == -1], [x[i][1] for i in range(n) if d[i] == -1], linestyle="none", marker="s", fillstyle="none")

    # Plot support vectors
    plt.plot([x[0] for x in xsv], [x[1] for x in xsv], color="k", marker="o", markersize=12, linestyle="none", fillstyle="none")

    # Evaluate the separator function on a eval_pts*eval_pts grid
    evals = np.zeros((eval_pts, eval_pts))
    for i in range(eval_pts):
        for j in range(eval_pts):
            evals[j][i] = sep(np.array([i / eval_pts, j / eval_pts]))
        print(i)

    # Plot H, H+ and H-
    xx, yy = np.meshgrid(np.linspace(0, 1, eval_pts), np.linspace(0, 1, eval_pts))
    contours = plt.contour(xx, yy, evals, levels=[-1, 0, 1])
    plt.clabel(contours)

    # Decorate plot
    plt.legend(["$C_{1}$", "$C_{-1}$", "Support vectors"])
    plt.title("Trained SVM separation")
    plt.savefig(fname="sep.pdf")

# Generate the training set
n = 100
x = [np.random.uniform(0, 1, size=2) for i in range(n)]
d = [1 if x[i][1] < 0.2 * np.sin(10 * x[i][0]) + 0.3 or (x[i][1] - 0.8) ** 2 + (x[i][0] - 0.5) ** 2 < 0.15 ** 2 else -1 for i in range(n)]

# Plot the training set
plot_data(n, x, d)

# Compute the separator
k1 = kern_poly(10)
k2 = kern_gaus(1)
sep, xsv, dsv = solve_svm(n, x, d, k2, 1e-6)

# Print support vectors
print("Support vectors:")
for sv in xsv:
    print(sv.transpose(), sep(sv))

# Evaluate the separator for drawing
eval_pts = 1000

# Plot the results
plot_sep(n, x, d, xsv, sep, eval_pts)