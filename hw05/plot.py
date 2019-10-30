import numpy as np
import matplotlib.pyplot as plt

# Use TeX for text rendering
plt.rc('text', usetex=True)

errors2 = [np.load("errors_10-10.npy"), np.load("errors_100-10.npy"), np.load("errors_200-10.npy")]
errors3 = [np.load("errors_10-10-10.npy"), np.load("errors_100-10-10.npy"), np.load("errors_100-100-10.npy")]
trainacc2 = [np.load("trainacc_10-10.npy"), np.load("trainacc_100-10.npy"), np.load("trainacc_200-10.npy")]
trainacc3 = [np.load("trainacc_10-10-10.npy"), np.load("trainacc_100-10-10.npy"), np.load("trainacc_100-100-10.npy")]
testacc2 = [np.load("testacc_10-10.npy"), np.load("testacc_100-10.npy"), np.load("testacc_200-10.npy")]
testacc3 = [np.load("testacc_10-10-10.npy"), np.load("testacc_100-10-10.npy"), np.load("testacc_100-100-10.npy")]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)
fig.suptitle("Error function (epochs 1-end)")

ax1.plot(range(1, len(errors2[0])), errors2[0][1:], marker="o", markersize=2)
ax1.plot(range(1, len(errors3[0])), errors3[0][1:], marker="o", markersize=2)
ax1.plot([-2, 102], [0.01, 0.01], linestyle="dashed")
ax1.legend(["Network \#1", "Network \#4", "0.01 threshold"])
ax1.set_title("Non-converging networks")

ax2.plot(range(1, len(errors2[1])), errors2[1][1:], marker="o", markersize=2)
ax2.plot(range(1, len(errors2[2])), errors2[2][1:], marker="o", markersize=2)
ax2.plot([-2, 70], [0.01, 0.01], linestyle="dashed")
ax2.legend(["Network \#2", "Network \#3", "0.01 threshold"])
ax2.set_title("2-layers converging networks")

ax3.plot(range(1, len(errors3[1])), errors3[1][1:], marker="o", markersize=2)
ax3.plot(range(1, len(errors3[2])), errors3[2][1:], marker="o", markersize=2)
ax3.plot([-2, 35], [0.01, 0.01], linestyle="dashed")
ax3.legend(["Network \#5", "Network \#6", "0.01 threshold"])
ax3.set_title("3-layers converging networks")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)
fig.suptitle("Test set accuracy (epochs 1-end)")

ax1.plot(range(1, len(testacc2[0])), testacc2[0][1:], marker="o", markersize=2)
ax1.plot(range(1, len(testacc3[0])), testacc3[0][1:], marker="o", markersize=2)
ax1.plot([-2, 102], [0.95, 0.95], linestyle="dashed")
ax1.legend(["Network \#1", "Network \#4", "95\% threshold"])
ax1.set_title("Non-converging networks")

ax2.plot(range(1, len(testacc2[1])), testacc2[1][1:], marker="o", markersize=2)
ax2.plot(range(1, len(testacc2[2])), testacc2[2][1:], marker="o", markersize=2)
ax2.plot([-2, 70], [0.95, 0.95], linestyle="dashed")
ax2.legend(["Network \#2", "Network \#3", "95\% threshold"])
ax2.set_title("2-layers converging networks")

ax3.plot(range(1, len(testacc3[1])), testacc3[1][1:], marker="o", markersize=2)
ax3.plot(range(1, len(testacc3[2])), testacc3[2][1:], marker="o", markersize=2)
ax3.plot([-2, 35], [0.95, 0.95], linestyle="dashed")
ax3.legend(["Network \#5", "Network \#6", "95\% threshold"])
ax3.set_title("3-layers converging networks")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)
fig.suptitle("Training set accuracy (epochs 1-end)")

ax1.plot(range(1, len(trainacc2[0])), trainacc2[0][1:], marker="o", markersize=2)
ax1.plot(range(1, len(trainacc3[0])), trainacc3[0][1:], marker="o", markersize=2)
ax1.plot([-2, 102], [0.95, 0.95], linestyle="dashed")
ax1.legend(["Network \#1", "Network \#4", "95\% threshold"])
ax1.set_title("Non-converging networks")

ax2.plot(range(1, len(trainacc2[1])), trainacc2[1][1:], marker="o", markersize=2)
ax2.plot(range(1, len(trainacc2[2])), trainacc2[2][1:], marker="o", markersize=2)
ax2.plot([-2, 70], [0.95, 0.95], linestyle="dashed")
ax2.legend(["Network \#2", "Network \#3", "95\% threshold"])
ax2.set_title("2-layers converging networks")

ax3.plot(range(1, len(trainacc3[1])), trainacc3[1][1:], marker="o", markersize=2)
ax3.plot(range(1, len(trainacc3[2])), trainacc3[2][1:], marker="o", markersize=2)
ax3.plot([-2, 35], [0.95, 0.95], linestyle="dashed")
ax3.legend(["Network \#5", "Network \#6", "95\% threshold"])
ax3.set_title("3-layers converging networks")

plt.show()