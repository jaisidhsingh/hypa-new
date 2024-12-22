import os
import sys
import json
import math
import torch
import wandb
import argparse
import warnings
from tqdm import tqdm
from contextlib import suppress
from torch.utils.data import DataLoader

from src.data import *
from src.models import *

from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs

from src.training import SeparateTrainer
from src.training.schedulers import cosine_lr
from src.training.loss_functions import ClipLoss

from src.utils import get_mapper_flops
from src.utils.backward_flops import FlopCounterMode
warnings.simplefilter("ignore")


def train_separate_mapper(args):
    # set the seed first
    torch.manual_seed(args.random_seed)

    # load in dataset for training
    train_dataset_config = data_configs.separate_embedding_dataset_configs(args)
    train_dataset = SeparateEmbeddings(train_dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=args.shuffle_data)
    print(f"Training data of {len(train_dataset)} samples loaded.")

    # the mapper
    model = MlpMapper(args.text_embed_dim, [], args.image_embed_dim, use_bias=args.use_bias, logit_scale=args.logit_scale)
    model = model.to(args.device)
    print("Mapper loaded.")

    # count FLOPs done in one step
    if args.flop_counter == "calflops":
        total_flops = get_mapper_flops(model, (args.batch_size, args.text_embed_dim), include_backward=True)
        total_flops = total_flops
        print(f"FLOPs per step = {total_flops[0]} x 10^({math.log10(total_flops[1])})")

    # CLIP settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = ClipLoss(args)

    # schedule learning rate and scale gradients
    total_steps = int(args.num_epochs * len(train_loader))
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    # trainer
    trainer = SeparateTrainer(args)
    logs = {}
    logs[f"configs"] = {
        "train_dataset": args.feature_dataset,
        "val_dataset": args.val_dataset,
        "test_dataset": args.test_dataset,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "steps_per_epoch": len(train_loader),
        "flops_per_epoch": total_flops[0] * total_flops[1]
    }

    # trackers
    bar = tqdm(total=args.num_epochs)

    # saving preparation
    ckpt_save_folder = os.path.join(args.checkpoint_folder, args.experiment_type, "separate", args.experiment_name, f"seed_{args.random_seed}")
    os.makedirs(ckpt_save_folder, exist_ok=True)

    # logs and results saving
    logs_save_folder = os.path.join(args.logs_folder, args.experiment_type, "separate", args.experiment_name, f"seed_{args.random_seed}")
    os.makedirs(logs_save_folder, exist_ok=True)

    if args.use_wandb:
        wandb.init(project="separate-mappers", name=args.experiment_name, entity="hyperalignment", config=vars(args))
    
    # training loop
    for epoch in range(args.num_epochs):
        train_logs, flop_count_results = trainer.train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch)

        if epoch == 0:
            print(flop_count_results)

        logs[f"epoch_{epoch+1}"] = {"train": train_logs}

        # update the progress bar
        bar.update(1)
        to_log = {
            "train_loss": train_logs["avg_loss"],
            "train_acc": train_logs["accuracy"],
        }
        bar.set_postfix(to_log)

        # save every some epochs for safety
        if (epoch+1) in [1, 2, 5, 10, 20, 40, 100, 200] and args.saving:
            dump = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "logs": logs,
                "one_epoch_flop_count": flop_count_results
            }

            torch.save(dump, os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}.pt"))
            tqdm.write(f"Checkpoint saved at epoch {epoch+1}.")

    bar.close()
    print("All done.")


if __name__ == "__main__":
    # experiment args
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--experiment-name", type=str, default="test")
    parser.add_argument("--experiment-type", type=str, default="multi_mapper")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--logs-folder", type=str, default="/home/mila/s/sparsha.mishra/projects/Hyper-Alignment/logs")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
    # model and dataset settings
    parser.add_argument("--image-encoder", type=str, default="flexivit_small.300ep_in1k")
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--feature-dataset", type=str, default="cc3m595k")
    parser.add_argument("--val-dataset", type=str, default="cc3mval_id_vitr_raw")
    parser.add_argument("--test-dataset", type=str, default="mscoco_val_id_vitr_var")
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--use-bias", type=bool, default=True)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    # training settings
    parser.add_argument("--batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--eval-batch-size", type=int, default=int(pow(2, 9)))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--shuffle-data", type=bool, default=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--saving", type=bool, default=True)
    parser.add_argument("--run-all-conds", type=bool, default=False)
    parser.add_argument("--flop-counter", type=str, default="custom")
    # get args object
    args = parser.parse_args()

    # models = [
    #     # "vit_base_patch16_224"
    #     # "vit_large_patch16_224.augreg_in21k_ft_in1k",
    #     # "vit_large_patch14_clip_336.laion2b_ft_in12k_in1k", #
    #     # "deit3_large_patch16_384.fb_in22k_ft_in1k", #
    #     # "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", #
    #     # "beit_large_patch16_384.in22k_ft_in22k_in1k", #
    #     # "beitv2_large_patch16_224.in1k_ft_in22k_in1k", #
    #     # "swin_base_patch4_window7_224.ms_in22k_ft_in1k", #
    #     # "convnext_base.fb_in22k_ft_in1k", # 
    #     # "convnextv2_base.fcmae_ft_in22k_in1k" #
    # ]
    # for model in models:
    #     args.image_encoder = model
    #     args.experiment_name = model
    #     train_separate_mapper(args)
    train_separate_mapper(args)
    print("All done.")
