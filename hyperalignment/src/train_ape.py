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
from torch.utils.flop_counter import FlopCounterMode

from data import *
from models import *
from models.param_decoders import MLP

from configs.data_configs import data_configs
from configs.model_configs import model_configs

from training import SeparateTrainer
from training.schedulers import cosine_lr
from training.loss_functions import ClipLoss
from evaluation import eval_classification, image_classification_eval
warnings.simplefilter("ignore")


@torch.no_grad()
def evaluate_mapper(args, model):
    model.eval()
    vlm = CustomVLM(args.image_encoder, args.text_encoder)
    vlm.mapper = model.to(args.device)
    acc, loss = eval_classification(args, vlm, vlm.image_encoder.transform, "imagenet1k")
    return acc, loss


def train_separate_mapper(args):
    # set the seed first
    torch.manual_seed(args.random_seed)

    # load in dataset for training
    train_dataset_config = data_configs.separate_embedding_dataset_configs(args)
    train_dataset = SeparateEmbeddings(train_dataset_config, split="train", args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    print(f"Training data of {len(train_dataset)} samples loaded.")

    # the connector
    model = MLP(args.text_embed_dim, [], args.image_embed_dim, use_bias=args.use_bias, logit_scale=args.logit_scale)
    model = model.to(args.device)
    print("Mapper loaded.")

    # CLIP settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = ClipLoss(args)

    # schedule learning rate and scale gradients
    total_steps = int(args.num_epochs * len(train_loader))
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.amp.autocast

    # trainer
    trainer = SeparateTrainer(args)
    logs = {}
    logs[f"configs"] = {
        "train_dataset": args.feature_dataset,
        "train_val_split_ratio": args.train_val_split_ratio,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "cosine_lr",
        "warmup_steps": args.warmup_steps,
        "num_epochs": args.num_epochs,
        "train_steps_per_epoch": len(train_loader),
    }

    # trackers
    bar = tqdm(total=args.num_epochs)
    flop_counter = FlopCounterMode(model, display=True, depth=2)

    # saving preparation
    ckpt_save_folder = os.path.join(args.checkpoint_folder, "ape", args.experiment_name, f"seed_{args.random_seed}")
    os.makedirs(ckpt_save_folder, exist_ok=True)

    # logs and results saving
    logs_save_folder = os.path.join(args.logs_folder, "ape", args.experiment_name, f"seed_{args.random_seed}")
    os.makedirs(logs_save_folder, exist_ok=True)

    if args.use_wandb:
        wandb.init(project="APE-training", name=args.experiment_name, entity="hyperalignment", config=vars(args))
    
    # training loop
    for epoch in range(args.num_epochs):
        # train for one epoch
        train_corrects, train_total = 0, 0
        train_running_loss = 0
        train_logs = {}
        model.train()

        # track training flops
        with flop_counter:
            for idx, (image_features, text_features) in enumerate(train_loader):
                step = int(epoch * len(train_loader)) + idx
                batch_size = image_features.shape[0]

                image_features = image_features.float().to(args.device)
                image_features = image_features.view(batch_size, args.image_embed_dim)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).to(args.device)

                text_features = text_features.float().to(args.device)
                text_features = text_features.view(batch_size, args.text_embed_dim)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(args.device)

                if scheduler is not None:
                    scheduler(step)

                optimizer.zero_grad()

                with autocast(args.device):
                    mapped_text_features = model(text_features)
                    mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True).to(args.device)
                    
                    sim = args.logit_scale * (image_features @ mapped_text_features.T)
                    labels = torch.arange(batch_size).long().to(args.device)
                    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

                in_batch_corrects = (sim.argmax(dim=-1) == labels).sum().item()
                train_running_loss += loss.item()
                train_corrects += in_batch_corrects
                train_total += labels.size(0)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        train_logs["flops"] = flop_counter.get_total_flops()
        train_logs["train_loss"] = train_running_loss / (idx+1)
        train_logs["train_accuracy"] = round(train_corrects/train_total * 100, 2)
        
        if args.use_wandb:
            wandb.log({"train_loss": train_running_loss / (idx+1), "train_accuracy": train_logs["train_accuracy"]}, step=epoch+1)

        # update the progress bar
        bar.update(1)
        to_log = {
            "train_loss": train_logs["train_loss"],
            "train_acc": train_logs["train_accuracy"],
            # "val_acc": val_acc,
            # "val_loss": val_loss,
        } 
        bar.set_postfix(to_log)
        logs[f"epoch_{epoch+1}"] = to_log

        # save every some epochs for safety
        if (epoch+1) in [1, 2, 5, 10, 20, 40, 100, 200] and args.saving:
            dump = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "logs": logs,
                "flops": train_logs["flops"]
            }

            torch.save(dump, os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}.pt"))
            with open(os.path.join(logs_save_folder, f"logs_{epoch+1}.json"), "w") as f:
                json.dump(logs, f)

            tqdm.write(f"Checkpoint saved at epoch {epoch+1}.")

    bar.close()
    print("All done.")
    return model
    # return model.layers[0].weight.data, model.layers[0].bias.data, val_acc


if __name__ == "__main__":
    # experiment args
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--experiment-name", type=str, default="vits_deit3s_in22k")
    parser.add_argument("--experiment-type", type=str, default="ape")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--logs-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/logs")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
    # model and dataset settings
    parser.add_argument("--image-encoder", type=str, default="deit3_small_patch16_224.fb_in22k_ft_in1k")
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--feature-dataset", type=str, default="cc3m558k")
    parser.add_argument("--val-dataset", type=str, default="cc3mval")
    parser.add_argument("--train-val-split-ratio", type=float, default=0.9)
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--use-bias", type=bool, default=True)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    parser.add_argument("--ablate", type=bool, default=False)
    # training settings
    parser.add_argument("--batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--eval-batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--shuffle-data", type=bool, default=False)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--saving", type=bool, default=True)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--data-scaling", type=float, default=1.0)
    # get args object
    args = parser.parse_args()
    model = train_separate_mapper(args)
    acc, loss = evaluate_mapper(args, model)
    print(acc)
