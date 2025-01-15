from pylab import plt

"""
ViT-L
"""
# ape_accs = [12.00, 38.40, 39.95]
# ape_epochs = [1, 5, 10]
# ape_flops = [2250.56e+9 * 37 * i for i in ape_epochs]

# hnet_acc = [31.45] #, 36.10]
# hnet_epochs = [1] #, 5]
# hnet_flops = [73151e+9 / 30]  #* i for i in hnet_epochs]

"""
ViT-S
"""
ape_accs = [12.0 , 20.5, 25.5]
ape_epochs = [1, 5, 10]
ape_flops = [30.484e+12 * i for i in ape_epochs]

hnet_acc = [10.7, 14.1, 23.9] #, 36.10]
hnet_epochs = [1, 5, 10] #, 5]
hnet_flops = [13.08e+12 / 12 * i for i in hnet_epochs]

"""

In one epoch for the hnet:
1. All samples once.
2. All models multiple times.

FLOPs_for_hnet_1ep = 23000 GFLOPs (num_ie = 12, all data samples in the dataset, regardless mini_batch_of_models)
input_to_hnet.shape = [len(models), 32]
input_passes_through_MLP_of_config = [128, 512, N*M]

FLOPs_for_APE_1ep =  31200 GFLOPs (num_ie = 1, per pair.)
input_to_APE_ll = [16384, 384]
input_passes_through_MLP_of_config = [768] (weights.shape = (384, 768))
"""

plt.plot(hnet_flops, hnet_acc, marker="*", label="H-Net ViT-S/16")
plt.plot(ape_flops, ape_accs, marker="o", label="APE ViT-S/16")
plt.xlabel("Trainable FLOPs")
# plt.xscale("log")
plt.ylabel("ImageNet-1k Top-1 Accuracy")
plt.ylim([0, 100])
plt.axhline(75.51, linestyle="--", c="black", label="CLIP ViT-S/14")
plt.legend()

for i in range(3):
    plt.annotate(f"{ape_epochs[i]}", (ape_flops[i], ape_accs[i] + 3))

    plt.annotate(f"{hnet_epochs[i]}", (hnet_flops[i], hnet_acc[i]))

plt.title("ViT-S/16 Performance v/s FLOPs tradeoff")
plt.savefig("new_flops_check.png")
