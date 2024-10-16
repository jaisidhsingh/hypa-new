import os
import sys
import json
import math
import torch
import argparse
import warnings
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree
import schedulefree

from src.data import *
from src.models import *

from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs

from src.training import SeparateTrainer
from src.training.schedulers import cosine_lr
from src.training.loss_functions import ClipLoss

from src.utils import get_mapper_flops
from src.utils.weight_analysis import WeightAnalyzer
warnings.simplefilter("ignore")


def train_separate_mapper(args):
    # set the seed first
    torch.manual_seed(args.random_seed)

    # load in dataset for training
    train_dataset_config = data_configs.separate_embedding_dataset_configs(args)
    train_dataset = SeparateEmbeddings(train_dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=args.shuffle_data)
    print(f"Training data of {len(train_dataset)} samples loaded.")

    # load in the jointly learnt mappers
    saved_params = torch.load(
        os.path.join(
            args.checkpoint_folder, args.experiment_type, "joint",
            args.ckpt_experiment_name, f"seed_{args.random_seed}", f"ckpt_{args.ckpt_epoch}.pt"
        )
    )["mapper_params"]

    experiment_config = model_configs.ID_experiment_configs[args.experiment_type]
    image_encoder_index = experiment_config["image_encoders"].index(args.image_encoder)
    text_encoder_index = experiment_config["text_encoders"].index(args.text_encoder)
    saved_weights = saved_params[image_encoder_index]

    # the mapper
    model = MlpMapper(args.text_embed_dim, [], args.image_embed_dim, use_bias=args.use_bias, logit_scale=args.logit_scale)
    model = model.to(args.device)
    model.layers[0].weight.data = saved_weights[0]
    model.layers[0].bias.data = saved_weights[1]
    print("Mapper loaded.")

    # count FLOPs done in one step
    total_flops = get_mapper_flops(model, (args.batch_size, args.text_embed_dim), include_backward=True)
    total_flops = total_flops
    print(f"FLOPs per step = {total_flops[0]} x 10^({math.log10(total_flops[1])})")

    # CLIP settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.learning_rate)
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
    weight_analysis_logs = {}

    # trackers
    bar = tqdm(total=args.num_epochs)
    best_epoch, best_accuracy, best_ckpt = 0, 0, None

    # saving preparation
    ckpt_save_folder = os.path.join(args.checkpoint_folder, args.experiment_type, "joint", args.ckpt_experiment_name, f"seed_{args.random_seed}", f"ie-{image_encoder_index}_te-{text_encoder_index}")
    os.makedirs(ckpt_save_folder, exist_ok=True)

    # logs and results saving
    logs_save_folder = os.path.join(args.logs_folder, args.experiment_type, "joint", args.ckpt_experiment_name, f"seed_{args.random_seed}", f"ie-{image_encoder_index}_te-{text_encoder_index}")
    os.makedirs(logs_save_folder, exist_ok=True)

    # training loop
    for epoch in range(args.num_epochs):
        train_logs = trainer.train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch)

        logs[f"epoch_{epoch+1}"] = {"train": train_logs}

        # update the progress bar
        bar.update(1)
        to_log = {
            "train_loss": train_logs["avg_loss"],
            "train_acc": train_logs["accuracy"],
        }
        bar.set_postfix(to_log)

        # save every some epochs for safety
        if (epoch+1) == args.num_epochs and args.saving:
            dump = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "logs": logs,
            }

            torch.save(dump, os.path.join(ckpt_save_folder, f"ft_ckpt_{epoch+1}.pt"))
            tqdm.write(f"Checkpoint saved at epoch {epoch+1}.")

            analyser = WeightAnalyzer(model)
            weight_analysis_data = {
                "esd_array" : analyser.esd.tolist(),
                "esd_hist_x" : analyser.esd_hist_x.tolist(),
                "esd_hist_y" : analyser.esd_hist_y.tolist(),
                "alpha": analyser.alpha,
				"lambda_xmin": analyser.lambda_xmin,
				"lambda_xmax": analyser.lambda_xmax,
            }
            weight_analysis_data.update({"summary": analyser.summary})
            weight_analysis_logs[f"epoch_{epoch+1}"] = weight_analysis_data

    bar.close()
    if args.saving:
        # save the logs in a json file
        with open(os.path.join(logs_save_folder, "ft_logs.json"), "w") as f:
            json.dump(logs, f)

        # save the weight analysis
        with open(os.path.join(logs_save_folder, "post_ft_weight_analysis_logs.json"), "w") as f:
            json.dump(weight_analysis_logs, f)



def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        args.random_seed = seed
        config = model_configs.ID_experiment_configs[args.experiment_type]
        for ie, image_encoder in enumerate(config["image_encoders"]):
            for te, text_encoder in enumerate(config["text_encoders"]):
                args.image_encoder = image_encoder
                args.text_encoder = text_encoder
                print(image_encoder, text_encoder)
                args.experiment_name = f"{args.experiment_type}_ie-{ie}_te-{te}.{args.feature_dataset}_epochs-{args.num_epochs}"
                print(args.experiment_name)
                train_separate_mapper(args)

    print("All done.")


if __name__ == "__main__":
    # experiment args
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--ckpt-experiment-name", type=str, default="hmlp-1_cond_emb-8.cc3m595k_epochs-1")
    parser.add_argument("--ckpt-epoch", type=int, default=1)
    parser.add_argument("--experiment-type", type=str, default="id_vitr")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--logs-folder", type=str, default="/home/mila/s/sparsha.mishra/projects/Hyper-Alignment/logs/id_vitr")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/id_vitr")
    # model and dataset settings
    parser.add_argument("--image-encoder", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--feature-dataset", type=str, default="cc3m595k_id_vitr_raw")
    parser.add_argument("--val-dataset", type=str, default="cc3mval_id_vitr_raw")
    parser.add_argument("--test-dataset", type=str, default="mscoco_val_id_vitr_var")
    parser.add_argument("--image-embed-dim", type=int, default=768)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--use-bias", type=bool, default=True)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    # training settings
    parser.add_argument("--batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--eval-batch-size", type=int, default=int(pow(2, 9)))
    parser.add_argument("--learning-rate", type=float, default=2e-1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--shuffle-data", type=bool, default=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--saving", type=bool, default=True)
    # get args object
    args = parser.parse_args()
    main(args)
