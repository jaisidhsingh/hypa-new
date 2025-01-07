import matplotlib.pyplot as plt
import numpy as np

models = ['ViT-S/16', 'ViT-B/16', 'ViT-L/16']
efficiency = [128.47, 256.9, 342.67]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(models, efficiency)
ax.set_xscale('log')
ax.set_xlabel('Training Efficiency')
ax.set_title('Model Training Efficiency Comparison')
ax.grid(True, which="both", ls="-", alpha=0.2)

# Add value annotations
for bar in bars:
    width = bar.get_width()
    ax.text(width*1.05, bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}', 
            va='center')

plt.tight_layout()
plt.savefig("small_scale_vit3_eff.png")
