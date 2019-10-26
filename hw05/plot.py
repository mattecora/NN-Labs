import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

errors = [np.load("errors_100-10.npy"), np.load("errors_200-10.npy"), np.load("errors_100-100-10.npy")]
test = [np.load("test_100-10.npy"), np.load("test_200-10.npy"), np.load("test_100-100-10.npy")]

plt.figure()
plt.plot(range(1, len(errors[0])), errors[0][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(errors[1])), errors[1][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(errors[2])), errors[2][1:], marker="o", markersize=5, fillstyle="none")
plt.plot([-2, 70], [0.01, 0.01], linestyle="dashed")
plt.gca().set_xlim(left=-2, right=70)
plt.legend(["Network \#1", "Network \#2", "Network \#3", "0.01 threshold"])
plt.title("Error function (epochs 1-end)")

plt.figure()
plt.plot(range(1, len(test[0])), test[0][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(test[1])), test[1][1:], marker="o", markersize=5, fillstyle="none")
plt.plot(range(1, len(test[2])), test[2][1:], marker="o", markersize=5, fillstyle="none")
plt.plot([-2, 70], [0.95, 0.95], linestyle="dashed")
plt.gca().set_xlim(left=-2, right=70)
plt.legend(["Network \#1", "Network \#2", "Network \#3", "95\% threshold"])
plt.title("Test set performance (epochs 1-end)")

plt.show()