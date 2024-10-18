import os
import clip
import timm
import wandb
import gdown
import argparse
import numpy as np
from tqdm import tqdm
from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from perturb import *
from src.models import MlpMapper
from src.training.schedulers import cosine_lr


class UnconditionalDataset(Dataset):
    def __init__(self, image_embeddings_path, text_embeddings_path, seeds):
        self.image_embeddings = np.load(image_embeddings_path)
        self.text_embeddings = np.load(text_embeddings_path)
        self.stats = get_embedding_stats

    def __len__(self):
        return self.image_embeddings.shape[0]

    def __getitem__(self, idx):
        image_embedding = torch.from_numpy(self.image_embeddings[idx, ...])
        text_embedding = torch.from_numpy(self.text_embeddings[idx, ...])
        return image_embedding, text_embedding
    

def main(args):
    model = MlpMapper(768, [], 384)
    model = model.to(args.device)

    seeds = [int(x) for x in args.seeds.split(",")]

    dataset = UnconditionalDataset(args.image_embeddings_path, args.text_embeddings_path, seeds)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.num_epochs * len(loader)

    if args.scheduler == "off":
        scheduler = None
    else:
        scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)

    scaler = torch.amp.GradScaler()
    autocast = torch.amp.autocast if args.device == "cuda" else suppress

    bar = tqdm(total=total_steps)
    logit_scale = torch.tensor(np.log(100.0)).to(args.device)

    if args.use_wandb:
        wandb.init(project="perturbations", entity="hyperalignment", config=vars(args))
    training_logs = {}

    ckpt_save_folder = "static_unconditional_checkpoints"
    os.makedirs(ckpt_save_folder, exist_ok=True)
    perturbations = get_perturbations_at_stats(dataset.stats, seeds)

    # training loop
    for epoch in range(args.num_epochs):
        correct, total = 0, 0
        running_loss = 0

        for idx, (image_embeddings, text_embeddings) in enumerate(loader):
            step = epoch * len(loader) + idx
            batch_size = image_embeddings.shape[0]
            image_embeddings = image_embeddings.to(args.device)

            if args.perturbation == "off":
                image_embeddings = image_embeddings.unsqueeze(1)
            elif args.perturbation == "static":
                image_embeddings = perturb_embeddings_statically(image_embeddings, seeds, perturbations)
            elif args.perturbation == "dynamic":
                shuffle_seed = step % args.shuffle_control 
                image_embeddings = perturb_embeddings_dynamically(image_embeddings, seeds, perturbations, shuffle_seed=shuffle_seed)

            text_embeddings = text_embeddings.to(args.device)
            labels = torch.arange(batch_size, dtype=torch.long).to(args.device)

            if scheduler is not None:
                scheduler(step)

            optimizer.zero_grad()

            with autocast():
                mapped_text_embeddings = model(image_embeddings)
                sim = logit_scale.exp() * torch.einsum("bnd,cd->nbc", image_embeddings, mapped_text_embeddings)
                correct += sim[0].argmax(dim=-1).eq(labels).sum().item()
                total += batch_size

                loss = 0
                for j in range(image_embeddings.shape[1]):
                    loss = loss + 0.5 * (F.cross_entropy(sim[j], labels) + F.cross_entropy(sim[j].T, labels))
                    loss = loss / image_embeddings.shape[1]

            accuracy = round(correct/total * 100, 2)
            running_loss = loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logs = {"loss": running_loss, "accuracy": accuracy}
            training_logs[f"epoch_{epoch+1}"] = logs

            wandb.log(logs, step=step)

            bar.set_postfix(logs)
            bar.set_description(f"Epoch {epoch+1}, step: {step+1}")
            bar.update(1)

        if epoch+1 in [1, 5, 10, 20, 40, 100, 200]:
            ckpt = {
                "model": model.state_dict(),
                "logs": training_logs
            }
            save_path = os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}.pt")
            torch.save(ckpt, save_path)

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-wandb", type=bool, default=False)
    parser.add_argument("--experiment-name", type=str, default="default_experiment")
    parser.add_argument("--checkpoint-folder", type=str, default="checkpoints")
    parser.add_argument("--image-encoder", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--feature-dataset", type=str, default="cc3m300k_id_vitr_var")
    parser.add_argument("--image-embed-dim", type=int, default=768)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
