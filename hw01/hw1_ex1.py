import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

# 2nd layer
S0 = [[-1, -1]]
S1 = [[-1, 1], [1, -1], [1, 1]]

plt.plot([x[0] for x in S0], [x[1] for x in S0], color="r", marker="s", linestyle="none", fillstyle="none")
plt.plot([x[0] for x in S1], [x[1] for x in S1], color="b", marker="o", linestyle="none", fillstyle="none")

xv = [-5, 5]
yv = [-x - 1 for x in xv]

plt.plot([-5, 5], [-x - 1 for x in [-5, 5]], color="g", linestyle="--")
plt.grid()
plt.axis([-2, 2, -2, 2])
plt.title("OR logical function using signum activation")
plt.legend(["$y = -1$", "$y = 1$"])
plt.show()