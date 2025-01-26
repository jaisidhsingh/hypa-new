import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from itertools import cycle
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.embedding_datasets import ImageEmbeddings, TextEmbeddings
from models.param_decoders import MLP
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from training.schedulers import cosine_lr
from train_ape import evaluate_mapper


def train_multi_ape(args):
    encoder_names = model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"][:args.num_image_encoders]

    image_loaders, image_datasets = {}, {}
    loader_len = 0
    for name in encoder_names:
        tmp_args = deepcopy(args)
        tmp_args.image_encoder = name
        image_datasets[name] = ImageEmbeddings(data_configs.separate_embedding_dataset_configs(tmp_args), split=None, args=None)
        image_loaders[name] = DataLoader(image_datasets[name], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        loader_len = len(image_loaders[name])
    
    for k, v in image_loaders.items():
        image_loaders[k] = cycle(v)
    
    text_dataset = TextEmbeddings(data_configs.separate_embedding_dataset_configs(args), split=None, args=None)
    text_loader = DataLoader(text_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    model = MLP(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = cosine_lr(optimizer, args.learning_rate, warmup_length=args.warmup_steps, steps=args.num_epochs * loader_len)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.amp.autocast

    bar = tqdm(total=args.num_epochs)
    logit_scale = torch.tensor(np.log(args.logit_scale)).to(args.device)
    
    ckpt_save_folder = os.path.join(args.checkpoint_folder, "multi_ape", args.experiment_name, f"seed_{args.random_seed}")
    os.makedirs(ckpt_save_folder, exist_ok=True)

    logs_save_folder = os.path.join(args.logs_folder, "multi_ape", args.experiment_name, f"seed_{args.random_seed}")
    os.makedirs(logs_save_folder, exist_ok=True)

    for epoch in range(args.num_epochs):
        corrects = {name: 0 for name in encoder_names}
        total = {name: 0 for name in encoder_names}
        accuracies = {name: 0 for name in encoder_names}

        for idx, text_embeddings in enumerate(text_loader):
            bs = len(text_embeddings)
            text_embeddings = text_embeddings.float().to(args.device)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

            image_embeddings = torch.cat([next(image_loaders[name]).unsqueeze(0) for name in encoder_names], dim=0).to(args.device)
            image_embeddings = image_embeddings.float().view(args.num_image_encoders, bs, args.image_embed_dim)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            step = epoch * loader_len + idx
            scheduler(step)
            optimizer.zero_grad()

            with autocast(args.device):
                mapped_text_embeddings = model(text_embeddings)
                mapped_text_embeddings = mapped_text_embeddings / mapped_text_embeddings.norm(dim=-1, keepdim=True)

                logits = torch.einsum("ncd,bd->ncb", image_embeddings, mapped_text_embeddings) * logit_scale
                labels = torch.arange(bs).long().to(args.device)

                loss = 0
                for j in range(args.num_image_encoders):
                    loss = loss + (F.cross_entropy(logits[j, :, :], labels) + F.cross_entropy(logits[j, :, :].T, labels)) / 2
                    corrects[encoder_names[j]] += (logits[j].argmax(dim=-1) == labels).sum().item()
                    total[encoder_names[j]] += bs
                
            loss = loss / args.num_image_encoders
            bar.set_postfix({"step": step, "total_loss": loss.item()})
            accuracies = {name: round(corrects[name] / total[name] * 100, 2) for name in encoder_names}
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        bar.update(1)
        
        if epoch+1 in [1, 2, 5, 10, 20, 40] and args.saving:
            dump = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(dump, os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}.pt"))
            tqdm.write(f"Checkpoint saved at epoch {epoch+1}.")
    
    bar.close()
    return ckpt_save_folder
        

if __name__ == "__main__":
    # experiment args
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--experiment-name", type=str, default="test")
    parser.add_argument("--experiment-type", type=str, default="multi_mapper")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--logs-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/logs")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
    # model and dataset settings
    parser.add_argument("--num-image-encoders", type=int, default=6)
    parser.add_argument("--image-encoder", type=str, default="vit_small_patch16_224")
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--shuffle-data", type=bool, default=False)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--saving", type=bool, default=False)
    parser.add_argument("--eval-every", type=int, default=1)
    # get args object
    args = parser.parse_args()

    args.num_epochs = 20
    args.saving = True

    # bs, lr = 16384, 1e-2
    bss = [int(pow(2, i)) for i in [8, 9, 10, 12, 14]]
    lrs = [1e-3, 1e-3, 3e-3, 5e-3, 1e-2]
    for bs, lr in zip(bss, lrs):
        args.experiment_name = f"vits_bs-{args.batch_size}_lr-{args.learning_rate}"
        args.batch_size = bs
        args.learning_rate = lr

        print(args.experiment_name, args.batch_size, args.learning_rate)
        f1 = train_multi_ape(args)
        
        res = {}
        ckpt = torch.load(os.path.join(f1, "ckpt_20.pt"))["model"]

        model = MLP(args.text_embed_dim, [], args.image_embed_dim, use_bias=args.use_bias, logit_scale=args.logit_scale)
        model.load_state_dict(ckpt)
        model.to(args.device)
        
        acc, loss = evaluate_mapper(args, model)
        res[args.experiment_name] = acc
        print(res)



    

