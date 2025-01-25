from pylab import plt

def one_tick(i):
    return r"$2^{" + str(i) + r"}$"
lbl = [one_tick(i) for i in [8, 9, 10, 12, 14]]
"""

ape_vits = {'vits_bs-256_lr-0.001': 25.45, 'vits_bs-1024_lr-0.003': 26.96, 'vits_bs-4096_lr-0.005': 27.7, 'vits_bs-16384_lr-0.01': 27.85}

plt.plot(bs, list(ape_vits.values()), marker='o')
plt.xticks(bs, [one_tick(i) for i in range(8, 15, 2)])

plt.xlabel("Batch size")
plt.ylabel("ImageNet-1k Top-1 Accuracy")
plt.title("APE: Batch size vs ImageNet-1k Top-1 Accuracy for ViT-S/16")
"""

# Data
flops = [877.7, 1097.2, 1536.1, 4161.8, 14675.9]  # FLOPs in billions
accuracies = [25.45, 26.27, 26.96, 27.70, 27.85]  # ImageNet-1k top-1 accuracy
# 'vits_bs-512_lr-0.001': 26.27
labels = [
    "bs 256, lr 1e-3",
    "bs 1024, lr 3e-3",
    "bs 4096, lr 5e-3",
    "bs 16384, lr 1e-2"
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(flops, accuracies, color='tab:blue', marker="o")  # Plot points
plt.xlabel('FLOPs (Billions)', fontsize=12)
plt.ylabel('ImageNet-1k Top-1 Accuracy', fontsize=12)
plt.title('APE: FLOPs vs ImageNet-1k Top-1 Accuracy\n(Training batch size reported)', fontsize=14)

# Add labels to each point
for i, label in enumerate(labels):
    plt.text(flops[i] - flops[0] / 4, accuracies[i], lbl[i], fontsize=12, ha='right')

# Show the plot
plt.grid(True)
# plt.show()
plt.savefig("ape_vits_bs.png")
