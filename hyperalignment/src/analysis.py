import os
import torch
import torch.nn.functional as F
from pylab import plt
import argparse
import warnings
import numpy as np
from src.utils.check_cka import CKA
from sklearn.metrics import mutual_info_score
warnings.simplefilter("ignore")

def compute_mutual_information(x, y, bins=20):
    out = np.zeros((x.shape[0], y.shape[0]))
    def int(x, y):
        x_binned = np.digitize(x, bins=np.histogram_bin_edges(x, bins=bins))
        y_binned = np.digitize(y, bins=np.histogram_bin_edges(y, bins=bins))
    
        mi = mutual_info_score(x_binned, y_binned)
        return mi
    
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            out[i][j] = int(x[i], y[j])

    return out

def load_experiment_data(args):
    path = os.path.join(args.ckpt_folder, args.experiment_name, f"seed_{args.random_seed}", f"ckpt_{args.ckpt_epoch}.pt")
    data = torch.load(path)
    mappers = data["mapper_params"]
    cond_embs = data["model"]["cond_embs.weight"]
    return {"mapper_weights": mappers, "cond_embs": cond_embs}

@torch.no_grad()
def cosine_sim(x1, x2):
    sim = F.normalize(x1, p=2, dim=1) @ F.normalize(x2, p=2, dim=1).T
    return sim

@torch.no_grad()
def euc_dist(x1, x2):
    return torch.cdist(x1, x2)

@torch.no_grad()
def cka(x1, x2):
    assert x1.device == x2.device
    d = x1.shape[1]
    cka = CKA(x1.device)
    out = torch.zeros((x1.shape[0], 2 * x2.shape[0]))

    for j in range(x1.shape[0]):
        for k in range(x2.shape[0]):
            out[j][k] = cka.linear_CKA(x1[j], x2[k])
        for l in range(x2.shape[0]):
            out[j][x2.shape[0] + l] = cka.kernel_CKA(x1[j], x2[l])

    return out

@torch.no_grad()
def plot_intra_side_by_side(args):
    data = load_experiment_data(args)
    print(f"Loaded data for experiment: {args.experiment_name}")

    group_size = int(args.experiment_name.split("_")[1]) // 3

    fig, axs = plt.subplots(3, 2, figsize=(args.plot_size, args.plot_size))
    for i in range(3):
        start = i * group_size
        end = i * group_size + group_size

        mapper_weights = data["mapper_weights"][start : end]

        c = 0
        for idx, item in enumerate(mapper_weights):
            if item[1].dim() == 1:
                mapper_weights[idx] = torch.cat([item[0], item[1].unsqueeze(-1)], dim=1)
                c += 1
        
        mapper_weights = torch.stack(mapper_weights)

        cond_embs = data["cond_embs"][start : end, ...].to(args.device)

        assert mapper_weights.shape[0] == cond_embs.shape[0]
        print(f"Prepared mapper weights and conditional embeddings for group {i+1}")

        setting_to_fn = {
            "cosine_sim": cosine_sim,
            "euc_dist": euc_dist,
            "cka": cka
        }
        mapper_grid = setting_to_fn[args.mapper_metric](mapper_weights, mapper_weights).cpu().numpy()
        cond_emb_grid = setting_to_fn[args.cond_emb_metric](cond_embs, cond_embs).cpu().numpy()
        cond_emb_mi = compute_mutual_information(cond_embs.cpu().numpy(), cond_embs.cpu().numpy())
        cond_emb_grid = np.concatenate([cond_emb_grid, cond_emb_mi], axis=1)

        axs[i, 0].imshow(mapper_grid)
        axs[i, 0].set_xlabel("Conn index")
        axs[i, 0].set_ylabel("Conn index")
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 0].set_title(f"Group {i+1} Conn\nLinear CKA -----|----- Kernel CKA")

        axs[i, 1].imshow(cond_emb_grid)
        axs[i, 1].set_xlabel("CE index")
        axs[i, 1].set_ylabel("CE index")
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        axs[i, 1].set_title(f"Group {i+1} CE\n{args.cond_emb_metric}----|----Mutual Inf.")
        print(f"Plotting done for group {i+1}")

    plt.tight_layout()

    os.makedirs(args.plot_save_folder, exist_ok=True)
    save_path = os.path.join(args.plot_save_folder, f"{args.plot_save_name}.pdf")
    
    plt.savefig(save_path, format="pdf")
    print(f"Plot saved at: {save_path}")


def plot_cond_emb_distr(args):
    data = load_experiment_data(args)
    print(f"Loaded data for experiment: {args.experiment_name}")

    group_size = int(args.experiment_name.split("_")[1]) // 3

    fig, axs = plt.subplots(1, 2, figsize=(args.plot_size, args.plot_size))
    ce_dim = int(args.experiment_name.split("_")[-1])

    cond_embs = data["cond_embs"]
    cossim = cosine_sim(cond_embs, cond_embs).cpu().numpy()
    mi = np.zeros()
 

    plt.tight_layout()
    os.makedirs(args.plot_save_folder, exist_ok=True)
    save_path = os.path.join(args.plot_save_folder, f"{args.plot_save_name}.pdf")
    
    plt.savefig(save_path, format="pdf")
    print(f"Plot saved at: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper")
    parser.add_argument("--experiment-name", type=str, default="ie_12_mlp_c_32")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--ckpt-epoch", type=int, default=1)
    parser.add_argument("--mapper-metric", type=str, default="cka", choices=["cosine_sim", "euc_dist", "cka"])
    parser.add_argument("--cond-emb-metric", type=str, default="cosine_sim", choices=["cosine_sim", "euc_dist", "cka"])
    parser.add_argument("--plot-save-folder", type=str, default="/home/mila/s/sparsha.mishra/projects/hypa-new/hyperalignment/plots")
    parser.add_argument("--plot-save-name", type=str, default="")
    parser.add_argument("--plot-size", type=int, default=20)

    args = parser.parse_args()
    args.plot_save_name = args.experiment_name + args.plot_save_name
    print("Start analysis for", args.experiment_name)
    plot_intra_side_by_side(args)
