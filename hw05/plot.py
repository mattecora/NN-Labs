import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

errors = [np.load("errors_10-10.npy"), np.load("errors_100-10.npy"), np.load("errors_200-10.npy"), np.load("errors_10-10-10.npy"), np.load("errors_100-10-10.npy"), np.load("errors_100-100-10.npy")]
trainacc = [np.load("trainacc_10-10.npy"), np.load("trainacc_100-10.npy"), np.load("trainacc_200-10.npy"), np.load("trainacc_10-10-10.npy"), np.load("trainacc_100-10-10.npy"), np.load("trainacc_100-100-10.npy")]
testacc = [np.load("testacc_10-10.npy"), np.load("testacc_100-10.npy"), np.load("testacc_200-10.npy"), np.load("testacc_10-10-10.npy"), np.load("testacc_100-10-10.npy"), np.load("testacc_100-100-10.npy")]

for i in range(len(errors)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)
    fig.suptitle(f"Performance of network \#{i+1} (epochs 1-end)")

    ax1.plot(range(1, len(trainacc[i])), trainacc[i][1:], marker="o", markersize=2)
    ax1.plot(range(1, len(testacc[i])), testacc[i][1:], marker="o", markersize=2)
    ax1.plot([-2, len(trainacc[i]) + 2], [0.95, 0.95], linestyle="dashed")
    ax1.set_title("Accuracy")
    ax1.legend(["Train set", "Test set", "95\% threshold"])

    ax2.plot(range(1, len(errors[i])), errors[i][1:], marker="o", markersize=2)
    ax2.plot([-2, len(errors[i]) + 2], [0.01, 0.01], linestyle="dashed")
    ax2.set_title("Error")
    ax2.legend(["Error", "0.01 threshold"])

plt.show()