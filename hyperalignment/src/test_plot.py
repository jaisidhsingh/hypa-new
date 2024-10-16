import matplotlib.pyplot as plt
import torch
import numpy as np


sep_epochs = [1, 5, 10]
joint_epochs = ["1", "FT-1"]
sep_flops = [58e+9 * 37 * i for i in sep_epochs]
joint_flops = [2.72e+9 * 1163, 2.72e+9 * 1163 + 58e+9 * 37 * 1]

ext = ["vanilla", "augreg", "mae", "dino", "openai", "rope"]

results = torch.load("logs/id_vitr/coco_eval_results.pt")
sep_accs = results["separate"]["mscoco"]["R@1"][:, 0, :3]
joint_accs = results["joint"]["mscoco"]["R@1"][:, 0, :2]

fig, axes = plt.subplots(2, 3)
for i in range(6):
    ax = axes[i//3, i%3]
    ax.plot(sep_flops, sep_accs[i], marker="o", label="APE")

    for j in range(3):
        ax.annotate(sep_epochs[j], (sep_flops[j], sep_accs[i][j]))

    ax.plot(joint_flops, joint_accs[i], marker="*", label="Ours")
    
    for j in range(2):
        ax.annotate(joint_epochs[j], (joint_flops[j], joint_accs[i][j]))

    ax.axhline(21.12, linestyle="--", label="CLIP ViT-B/16", color="black")

    ax.set_ylim([0, 50])
    ax.set_ylabel("R@1")
    ax.set_xlabel("FLOPs")
    ax.set_title(ext[i])

plt.legend()
plt.tight_layout()
plt.show()

# plt.plot(sep_flops, sep_accs, marker="o", label="APE")
# plt.plot(joint_flops, joint_accs, marker="*", linestyle="--", label="Our")
# plt.legend()
# plt.show()
