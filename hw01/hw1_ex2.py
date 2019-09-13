import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

# First graph
fig, (plt1, plt2, plt3) = plt.subplots(1, 3)

plt1.axis([-2.5, 2.5, -2.5, 2.5])
plt2.axis([-2.5, 2.5, -2.5, 2.5])
plt3.axis([-2.5, 2.5, -2.5, 2.5])

plt1.axis("scaled")
plt2.axis("scaled")
plt3.axis("scaled")

plt1.title.set_text("Output of neuron 1")
plt2.title.set_text("Output of neuron 2")
plt3.title.set_text("Output of neuron 3")

# 1st layer, 1st neuron: 1 + x - y >= 0 -> y <= x + 1
plt1.plot([-5, 5], [x + 1 for x in [-5, 5]], color="r")
plt1.fill_between([-5, 5], [x + 1 for x in [-5, 5]], -100, color="r", alpha=0.25)

# 1st layer, 2nd neuron: 1 - x - y >= 0 -> y <= -x + 1
plt2.plot([-5, 5], [-x + 1 for x in [-5, 5]], color="b")
plt2.fill_between([-5, 5], [-x + 1 for x in [-5, 5]], -100, color="b", alpha=0.25)

# 1st layer, 3rd neuron: -x >= 0 -> x <= 0
plt3.plot([0, 0], [-5, 5], color="g")
plt3.fill_betweenx([-5, 5], [0, 0], -100, color="g", alpha=0.25)

plt.draw()

# Second graph
plt.figure()
plt.axis([-2.5, 2.5, -2.5, 2.5])
plt.title("Combination of output regions of first layer neurons")

# 1st layer, 1st neuron: 1 + x - y >= 0 -> y <= x + 1
plt.plot([-5, 5], [x + 1 for x in [-5, 5]], color="r")
plt.fill_between([-5, 5], [x + 1 for x in [-5, 5]], -100, color="r", alpha=0.25)

# 1st layer, 2nd neuron: 1 - x - y >= 0 -> y <= -x + 1
plt.plot([-5, 5], [-x + 1 for x in [-5, 5]], color="b")
plt.fill_between([-5, 5], [-x + 1 for x in [-5, 5]], -100, color="b", alpha=0.25)

# 1st layer, 3rd neuron: -x >= 0 -> x <= 0
plt.plot([0, 0], [-5, 5], color="g")
plt.fill_betweenx([-5, 5], [0, 0], -100, color="g", alpha=0.25)

# Labels
# plt.text(0.35, 2, "$y_1 = 0$ \n $y_2 = 0$ \n $y_3 = 0$", fontsize=12, ha="center", va="center")
# plt.text(1.5, 1, "$y_1 = 1$ \n $y_2 = 0$ \n $y_3 = 0$", fontsize=12, ha="center", va="center")
# plt.text(1, -1.5, "$y_1 = 1$ \n $y_2 = 1$ \n $y_3 = 0$", fontsize=12, ha="center", va="center")
# plt.text(-1, -1.5, "$y_1 = 1$ \n $y_2 = 1$ \n $y_3 = 1$", fontsize=12, ha="center", va="center")
# plt.text(-1.5, 1, "$y_1 = 0$ \n $y_2 = 1$ \n $y_3 = 1$", fontsize=12, ha="center", va="center")
# plt.text(-0.35, 2, "$y_1 = 0$ \n $y_2 = 0$ \n $y_3 = 1$", fontsize=12, ha="center", va="center")

plt.text(0.35, 2, "1", fontsize=18, ha="center", va="center")
plt.text(1.5, 1, "2", fontsize=18, ha="center", va="center")
plt.text(1, -1.5, "3", fontsize=18, ha="center", va="center")
plt.text(-1, -1.5, "4", fontsize=18, ha="center", va="center")
plt.text(-1.5, 1, "5", fontsize=18, ha="center", va="center")
plt.text(-0.35, 2, "6", fontsize=18, ha="center", va="center")

# Second graph
plt.figure()
plt.axis([-2.5, 2.5, -2.5, 2.5])
plt.title("Classes separation using the given network")

plt.plot([0, 5], [-x + 1 for x in [0, 5]], color="black")
plt.plot([0, 0], [-5, 1], color="black", linestyle="--")

plt.fill_betweenx([-x + 1 for x in [0, 5]], [0,5], color="lightgray")
plt.text(1, -1.5, "$z = 1$", fontsize=18, ha="center", va="center")
plt.text(-1, 1.5, "$z = 0$", fontsize=18, ha="center", va="center")

plt.draw()
plt.show()