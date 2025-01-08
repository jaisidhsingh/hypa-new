from pylab import plt

ape_accs = [12.00, 38.40, 39.95]
ape_epochs = [1, 5, 10]
ape_flops = [2250.56e+9 * 37 * i for i in ape_epochs]

hnet_acc = [31.45, 36.10]
hnet_epochs = [1, 5]
hnet_flops = [73151e+9 * i for i in hnet_epochs]

plt.plot(hnet_flops, hnet_acc, marker="*", label="H-Net ViT-L/16")
plt.plot(ape_flops, ape_accs, marker="o", label="APE ViT-L/16")
plt.xlabel("Trainable FLOPs")
# plt.xscale("log")
plt.ylabel("ImageNet-1k Top-1 Accuracy")
plt.ylim([0, 100])
plt.axhline(75.51, linestyle="--", c="black", label="CLIP ViT-L/14")
plt.legend()

for i in range(3):
    plt.annotate(str(ape_epochs[i]), (ape_flops[i], ape_accs[i]))

plt.annotate(1, (hnet_flops[0], hnet_acc[0]))

plt.savefig("vitl_imagenet_flops.png")
