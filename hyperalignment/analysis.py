import os
import torch
import torch.nn.functional as F
from pylab import plt
import argparse
import warnings
from src.utils.check_cka import CKA
warnings.simplefilter("ignore")


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
        # mapper_weights = mapper_weights.view(mapper_weights.shape[0], -1).to(args.device)

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
        axs[i, 1].set_title(f"Group {i+1} CE\n{args.cond_emb_metric}")
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

    fig, axs = plt.subplots(3, 2, figsize=(args.plot_size, args.plot_size))
    ce_dim = int(args.experiment_name.split("_")[-1])

    for i in range(3):
        cond_embs = data["cond_embs"]
        l1_norms = [torch.norm(ce, p=1).item() for ce in cond_embs]

        axs[i, 0].imshow(cond_embs.cpu().numpy())
        axs[i, 0].set_ylabel("CE index")
        axs[i, 0].set_xlabel("CE dim =", ce_dim)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 0].set_title(f"Group {i+1} CEs")

        axs[i, 1].plot([j for j in range(len(l1_norms))], l1_norms)
        axs[i, 1].set_xlabel("CE index")
        axs[i, 1].set_ylabel("L1 norm")
        axs[i, 1].set_title(f"Group {i+1} CE L1 norms")
        axs[i, 1].grid()
        axs[i, 1].set_xticks([j for j in range(len(l1_norms))])

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
    parser.add_argument("--plot-save-name", type=str, default="_ce_dist")
    parser.add_argument("--plot-size", type=int, default=20)

    args = parser.parse_args()
    args.plot_save_name = args.experiment_name + args.plot_save_name
    plot_cond_emb_distr(args)
