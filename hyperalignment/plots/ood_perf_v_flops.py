from pylab import plt

models = ["H-Net mini", "H-Net full", "APE full"]
values = [23.0, 25.8, 23.8]

bars = plt.bar(models, values, width=0.4)
bars[0].set_color("tab:purple")
bars[2].set_color("tab:orange")

plt.ylabel("ImageNet-1k Top-1 Accuracy (%)")
plt.xlabel("Methods")
plt.show()