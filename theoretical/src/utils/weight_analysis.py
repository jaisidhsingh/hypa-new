import torch
import os
import warnings
import numpy as np
import weightwatcher as ww
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")


class WeightAnalyzer():
    def __init__(self, model, esd_bins=100):
        self.model = model
        self.esd_bins = esd_bins

        self.watcher = ww.WeightWatcher(model=model)
        self.details = self.watcher.analyze()
        self.summary = self.watcher.get_summary(self.details)
        
        self.esd = self.watcher.get_ESD()
        hist, bin_edges = np.histogram(self.esd, bins=self.esd_bins)
        self.esd_hist_x = bin_edges
        self.esd_hist_y = hist

        self.alpha = round(float(self.details["alpha"].iloc[0]), 2)
        self.lambda_xmin = round(float(self.details["xmin"].iloc[0]), 2)
        self.lambda_xmax = round(float(self.details["xmax"].iloc[0]), 2)

    def plot_esd(self, save_path):
        hist, bin_edges = np.histogram(self.watcher.get_ESD(), bins=self.esd_bins)
        scales = ["lin-lin", "log-log"]

        fig, axes = plt.subplots(1, len(scales))
        for idx, scale in enumerate(scales):
            axes[idx].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7)
            axes[idx].axvline(self.lambda_xmin, color="red", label=r"$\lambda_{min}$")
            axes[idx].axvline(self.lambda_xmax, color="purple", label=r"$\lambda_{max}$")
            
            if idx == 0:
                axes[idx].legend()

            if scale == "log-log":
                axes[idx].set_xscale("log")
                axes[idx].set_yscale("log")
                pl_fit_x = [self.lambda_xmin, self.lambda_xmax]
                pl_fit_y = [self.alpha, hist[-1]]

            axes[idx].set_xlim([0.0, 2.0])
            axes[idx].set_xlabel("Eigenvalues")
            axes[idx].set_ylabel("Frequency")

        title_intr = f"ESD plot {scale}"
        title_alpha = r"$\alpha=$" + str(round(self.alpha, 3)) + " , " + r"$\lambda_{min}=$" + str(round(self.lambda_xmin, 2)) + " , " + r"$\lambda_{max}=$" + str(round(self.lambda_xmax, 2))
        fig.suptitle(f"{title_intr}\n{title_alpha}")
        plt.tight_layout()

        save_path = os.path.join(f"{self.experiment_name}_esd.jpg")
        plt.savefig(save_path, dpi=300)

