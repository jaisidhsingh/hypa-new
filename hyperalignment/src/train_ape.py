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
warnings.simplefilter("ignore")


def train_separate_mapper(args):
    # set the seed first
    torch.manual_seed(args.random_seed)

    # load in dataset for training
    train_dataset_config = data_configs.separate_embedding_dataset_configs(args)
    train_dataset = SeparateEmbeddings(train_dataset_config, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=args.shuffle_data)
    print(f"Training data of {len(train_dataset)} samples loaded.")

    # load in dataset for validation
    val_dataset = SeparateEmbeddings(train_dataset_config, split="val", split_ratio=args.train_val_split_ratio)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=args.shuffle_data)
    print(f"Validation data of {len(val_dataset)} samples loaded.")

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

                image_features = image_features.float()
                image_features = image_features.view(batch_size, args.image_embed_dim)

                text_features = text_features.float().to(args.device)
                text_features = text_features.view(batch_size, args.text_embed_dim)

                if scheduler is not None:
                    scheduler(step)

                optimizer.zero_grad()

                with autocast(args.device):
                    mapped_text_features = model(text_features)
                    mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)
                    
                    sim = model.logit_scale.exp().to(args.device) * (image_features @ mapped_text_features.T)
                    labels = torch.arange(batch_size).long().to(args.device)
                    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


                in_batch_corrects = (sim.argmax(dim=-1) == labels).sum().item()
                print(in_batch_corrects)            
                train_running_loss += loss.item()
                train_corrects += in_batch_corrects
                train_total += batch_size
                train_accuracy = round(train_corrects/train_total * 100, 2)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100))
                    
                del image_features
                del text_features
                del mapped_text_features
        
        train_logs["flops"] = flop_counter.get_total_flops()
        train_logs["train_loss"] = train_running_loss / (idx+1)
        train_logs["train_accuracy"] = train_accuracy
        
        if args.use_wandb:
            wandb.log({"train_loss": train_running_loss / (idx+1), "train_accuracy": train_accuracy}, step=epoch+1)
        
        print(train_corrects, train_total)

        # now validate
        val_corrects, val_total = 0, 0
        val_running_loss = 0
        val_logs = {}
        model.eval()

        with torch.no_grad():
            for (image_features, text_features) in val_loader:
                batch_size = image_features.shape[0]

                image_features = image_features.float()
                image_features = image_features.view(batch_size, args.image_embed_dim)
                
                text_features = text_features.float().to(args.device)
                text_features = text_features.view(batch_size, args.text_embed_dim)

                # with autocast(args.device):
                mapped_text_features = model(text_features)
                mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)

                sim = model.logit_scale.exp().to(args.device) * (image_features @ mapped_text_features.T)
                labels = torch.arange(batch_size).long().to(args.device)
                loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

                in_batch_corrects = (sim.argmax(dim=-1) == labels).sum().item()
                val_running_loss += loss.item()
                val_corrects += in_batch_corrects
                val_total += batch_size
                val_accuracy = round(val_corrects/val_total * 100, 2)
                
                del image_features
                del text_features
                del mapped_text_features
        
        val_logs["val_loss"] = val_running_loss / len(val_loader)
        val_logs["val_accuracy"] = val_accuracy
        logs[f"epoch_{epoch+1}"] = {"train": train_logs, "val": val_logs}

        print(val_corrects, val_total)
        
        if args.use_wandb:
            wandb.log({"val_loss": val_running_loss / len(val_loader), "val_accuracy": val_accuracy}, step=epoch+1)  


        # update the progress bar
        bar.update(1)
        to_log = {
            "train_loss": train_logs["train_loss"],
            "train_acc": train_logs["train_accuracy"],
            "val_loss": val_logs["val_loss"],
            "val_acc": val_logs["val_accuracy"]
        }
        bar.set_postfix(to_log)

        # save every some epochs for safety
        if (epoch+1) in [1, 2, 5, 10, 20, 40, 100, 200] and args.saving:
            dump = {
                "model": model.state_dict(),
                "logs": logs,
            }
            # "optimizer": optimizer.state_dict(),
            # "one_epoch_flop_count": train_logs["flops"]

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
    parser.add_argument("--image-encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--feature-dataset", type=str, default="cc3m595k")
    parser.add_argument("--val-dataset", type=str, default="cc3mval")
    parser.add_argument("--train-val-split-ratio", type=float, default=0.9)
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--use-bias", type=bool, default=True)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    # training settings
    parser.add_argument("--batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--eval-batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--shuffle-data", type=bool, default=False)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--saving", type=bool, default=False)
    # get args object
    args = parser.parse_args()

    suffix = f"bs-{args.batch_size}_lr-{args.learning_rate}_ep-{args.num_epochs}"
    args.experiment_name = f"{args.image_encoder}_{args.text_encoder}_{suffix}"
    train_separate_mapper(args)
    print("All done.") 

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
    # args.experiment_name = args.image_encoder
    #     train_separate_mapper(args)
    # train_separate_mapper(args)
    # print("All done.")
