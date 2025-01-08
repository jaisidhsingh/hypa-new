import numpy as np
import matplotlib.pyplot as plt

models = ['ViT-S/16', 'ViT-B/16', 'ViT-L/16']

# APE (baseline) data
ape_cifar10 = [90.1, 91.4, 96.6]
ape_cifar100 = [51.9, 63.0, 71.2]
ape_imagenet = [25.5, 38.9, 40.0]

# H-Net data
hnet_cifar10 = [89.8, 93.6, 94.3]
hnet_cifar100 = [46.7, 58.0, 62.0]
hnet_imagenet = [23.4, 30.3, 31.4]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
width = 0.35
x = np.arange(len(models))

def create_bar_plot(ax, ape, hnet, clip, title):
    ax.bar(x - width/2, hnet, width, label='H-Net')
    ax.bar(x + width/2, ape, width, label='APE')
    ax.axhline(clip, linestyle="--", c="black", label="CLIP ViT-L/14")
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylim([0, 100])
    ax.legend(loc="lower right")

create_bar_plot(ax1, ape_cifar10, hnet_cifar10, 91.40, 'CIFAR-10')
create_bar_plot(ax2, ape_cifar100, hnet_cifar100, 65.95, 'CIFAR-100')
create_bar_plot(ax3, ape_imagenet, hnet_imagenet, 75.51, 'ImageNet-1k')

plt.tight_layout()
plt.savefig("small_scale_vits_img_clf.png")
