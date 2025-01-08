from pylab import plt

models = ["H-Net OOD", "APE"]
values = [14.6, 2.29]

bars = plt.bar(models, values, width=0.4)
bars[0].set_color("tab:purple")
bars[1].set_color("tab:orange")

plt.ylabel("ImageNet-1k Top-1 Accuracy (%)")
plt.xlabel("Method")
plt.show()


eff_names = ["H-Net mini", "H-Net full"]
eff_values = [20.0 , 1.73]
ebars = plt.bar(eff_names, eff_values, width=0.6)
bars[0].set_color("tab:purple")
plt.ylabel("Efficiency in OOD over APE")
plt.xlabel("Method")
plt.show()