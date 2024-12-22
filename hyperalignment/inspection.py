import os
import torch

from src.utils.check_cka import CKA


def load_mm_ckpt(args):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper"
    path = os.path.join(folder, args.exp_name, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    ckpt = torch.load(path)
    