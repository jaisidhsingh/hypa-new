from pylab import plt

models = ["H-Net OOD", "APE"]
values = [0.12, 2.29]

bars = plt.bar(models, values, width=0.4)
bars[0].set_color("tab:purple")
bars[1].set_color("tab:orange")

plt.ylabel("ImageNet-1k Top-1 Accuracy (%)")
plt.xlabel("Method")
plt.title("Our OOD opt. v/s APE (at 31.2 GFLOPs budget for both)")
plt.savefig("results/ood_flexivit_ape_sb.png")
