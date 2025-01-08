from pylab import plt

models = ["H-Net mini", "H-Net full", "APE full"]
values = [0.0, 25.8, 23.8]

plt.bar(models, values)
plt.ylabel("ImageNet-1k Top-1 Accuracy (%)")
plt.xlabel("Methods")
plt.show()