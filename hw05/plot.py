import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

errors = [np.load("errors_100-10.npy"), np.load("errors_200-10.npy"), np.load("errors_100-100-10.npy")]
trainacc = [np.load("trainacc_100-10.npy"), np.load("trainacc_200-10.npy"), np.load("trainacc_100-100-10.npy")]
testacc = [np.load("testacc_100-10.npy"), np.load("testacc_200-10.npy"), np.load("testacc_100-100-10.npy")]

plt.figure()
plt.plot(range(1, len(errors[0])), errors[0][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(errors[1])), errors[1][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(errors[2])), errors[2][1:], marker="o", markersize=5, fillstyle="none")
plt.plot([-2, 70], [0.01, 0.01], linestyle="dashed")
plt.gca().set_xlim(left=-2, right=70)
plt.legend(["Network \#1", "Network \#2", "Network \#3", "0.01 threshold"])
plt.title("Error function (epochs 1-end)")

plt.figure()
plt.plot(range(1, len(testacc[0])), testacc[0][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(testacc[1])), testacc[1][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(testacc[2])), testacc[2][1:], marker="o", markersize=5, fillstyle="none")
plt.plot([-2, 70], [0.95, 0.95], linestyle="dashed")
plt.gca().set_xlim(left=-2, right=70)
plt.legend(["Network \#1", "Network \#2", "Network \#3", "95\% threshold"])
plt.title("Test set performance (epochs 1-end)")

plt.figure()
plt.plot(range(1, len(trainacc[0])), trainacc[0][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(trainacc[1])), trainacc[1][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(trainacc[2])), trainacc[2][1:], marker="o", markersize=5, fillstyle="none")
plt.plot([-2, 70], [0.95, 0.95], linestyle="dashed")
plt.gca().set_xlim(left=-2, right=70)
plt.legend(["Network \#1", "Network \#2", "Network \#3", "95\% threshold"])
plt.title("Training set performance (epochs 1-end)")

plt.show()