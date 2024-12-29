import sys
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from src.models import *
from src.training.schedulers import *


def rescale_vector(x, target_min=1e-3, target_max=1e-2):
    x_min = x.min()
    x_max = x.max()
    
    normalized = (x - x_min) / (x_max - x_min)
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled


def get_onhand_std(root_folder, scale_factor=1e-1):
    encoders = os.listdir(root_folder)
    paths = [os.path.join(root_folder, enc, "memmap.npy") for enc in encoders if ".npy" not in enc]

    memmaps = [np.memmap(path, dtype="float32", mode="r", shape=(595375, 384)) for path in tqdm(paths)] # each has shape: L x D
    chunk_size = 50000
    chunk_means = []
    for i in tqdm(range(0, 595375, 50000)):
        idx = min(i + chunk_size, 595375)

        catted = np.concatenate(
            [ np.expand_dims(mm[i:idx, :], axis=1) for mm in memmaps ],
            axis=1
        )
        chunk_means.append((idx-i, catted.std(axis=1).mean(axis=0)))
    
    x = np.zeros((384,))
    for item in chunk_means:
        x += item[0] * item[1]
    
    x /= 595375
    out = torch.from_numpy(x)

    if scale_factor is not None:
        out *= scale_factor
    
    return out


def perturb_std(std, seed, scale_factor=1e-4):
    torch.manual_seed(seed)
    noise = torch.randn(std.shape)
    scaled_noise = noise
    if scale_factor is not None:
        scaled_noise *= 1e-4

    perturbed_std = std + scaled_noise
    return perturbed_std


def get_multiseed_perturbations(seeds, std):
    output = []
    mean = torch.zeros(std.shape)
    for seed in seeds:
        perturbed_std = perturb_std(std, seed)
        embedding_perturbation = torch.normal(mean, perturbed_std)
        output.append(embedding_perturbation)
    
    return output 


def perturb_embeddings(embeddings, perturbations):
    embed_dim = embeddings.shape[1]
    batch_size = embeddings.shape[0]

    output = torch.zeros((embeddings.shape[0], len(perturbations)+1, embed_dim))
    output[:, 0, :] = embeddings
    for i in range(1, len(perturbations)+1):
        new_embeddings = embeddings + perturbations[i-1].repeat(batch_size, 1)
        output[:, i, :] = new_embeddings
    
    return output


def shuffle_perturbations(perturbation, seed):
    torch.manual_seed(seed)
    idx = torch.randperm(perturbation.shape[-1])
    shuffled_perturbation = perturbation[idx]
    return shuffled_perturbation


def dynamic_perturbation(embeddings, perturbations, shuffle_seed):
    # in place modifier 
    batch_size = embeddings.shape[0]

    out = torch.zeros((batch_size, len(perturbations)+1, embeddings.shape[-1])).to(embeddings.device)
    out[:, 0, :] = embeddings
    for i in range(1, len(perturbations)+1):
        shuffled_perturbation = shuffle_perturbations(perturbations[i-1], shuffle_seed)
        new_embeddings = embeddings + shuffled_perturbation.repeat(batch_size, 1).to(embeddings.device)
        out[:, i, :] = new_embeddings
    
    return out


class DynamicUnconDataset(Dataset):
    def __init__(self, image_embeddings_path, text_embeddings_path):
        self.image_embeddings = np.load(image_embeddings_path)
        self.text_embeddings = np.load(text_embeddings_path)
        self.num_samples = self.image_embeddings.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.image_embeddings[idx, :]), torch.from_numpy(self.text_embeddings[idx, :])


class LossScaler(torch.nn.Module):
    def __init__(self, num_scalars):
        super().__init__()
        self.num_scalars = num_scalars
        self.loss_coefficients = torch.nn.Parameter(torch.randn(num_scalars))
    
    def forward(self):
        return F.softmax(self.loss_coefficients, dim=0)


def init_dynamic_loader(args):
    seeds = [int(x) for x in args.dynamic_seeds.split(",")]
    while True:
        for seed in seeds:
            yield seed


def main(args, case_num, perturbations):
    image_root_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_384"
    text_root_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/text_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_768"

    image_embeddings_path = os.path.join(image_root_folder, "vit_small_patch16_224","embeddings.npy")
    text_embeddings_path = os.path.join(text_root_folder, "sentence-t5-base","embeddings.npy")

    dataset = DynamicUnconDataset(image_embeddings_path, text_embeddings_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    model = MlpMapper(768, [], 384)
    model = model.to(args.device)

    loss_coefficients = None
    if args.loss_scaling == "learnt_scalar":
        loss_coefficients = LossScaler(case_num).to(args.device) 

    if loss_coefficients is not None:
        parameters = [p for p in model.parameters()]
        for param in loss_coefficients.parameters():
            parameters.append(param)
    else:
        parameters = [p for p in model.parameters()]

    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.num_epochs * len(loader)

    if args.scheduler == "off":
        scheduler = None
    else:
        scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    bar = tqdm(total=int(args.num_epochs * len(loader)))
    logs = {}
    logit_scale = torch.tensor(np.log(100.0)).to(args.device)
    dynamic_loader = init_dynamic_loader(args)

    ckpt_save_folder = "/home/mila/s/sparsha.mishra/scratch/hypa/checkpoints/uncond/vit_s16_with_5_shuffled"
    os.makedirs(ckpt_save_folder, exist_ok=True)

    for epoch in range(args.num_epochs):
        correct, total = 0, 0
        store = {"loss": [], "accuracy": []}

        running_loss = 0
        for idx, (image_features, text_features) in enumerate(loader):
            image_features = image_features.float().to(args.device)
            text_features = text_features.float().to(args.device)

            # now make the perturbations change over time
            dynamic_seed = next(dynamic_loader)
            image_features = dynamic_perturbation(image_features, perturbations, dynamic_seed)
            
            batch_size = image_features.shape[0]
            labels = torch.arange(batch_size, dtype=torch.long).to(args.device)

            step = epoch * len(loader) + idx
            if scheduler is not None:
                scheduler(step)

            optimizer.zero_grad()

            with autocast():
                mapped_text_features = model(text_features)
                if args.loss_scaling == "learnt_scalar":
                    loss_scales = loss_coefficients()
                sim = logit_scale.exp() * torch.einsum("bnd,cd->nbc", image_features, mapped_text_features)
                loss = 0
                for j in range(case_num):
                    loss = loss + F.cross_entropy(sim[j], labels) + F.cross_entropy(sim[j].T, labels)
                    loss = loss * 0.5
                    if args.loss_scaling == "average":
                        loss = loss / case_num
                    elif args.loss_scaling == "learnt_scalar":
                        loss = loss * loss_scales[j]
                
            running_loss = loss.item()
            store["loss"].append(running_loss)

            correct += (sim[0].argmax(dim=-1) == labels).sum().item()
            total += batch_size
            accuracy = round(correct/total * 100, 2)
            store["accuracy"].append(accuracy)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            in_step_logs = {"avg_loss": running_loss, "accuracy_0": accuracy}
            if case_num == 1:
                in_step_logs["loss"] = running_loss
                in_step_logs.pop("avg_loss")
            
            bar.set_postfix(in_step_logs)
            bar.set_description(f"Epoch: {epoch+1}")
            bar.update(1)
        
        logs[f"epoch_{epoch+1}"] = store 

        if (epoch+1) in [1, 5, 10, 20, 40, 100, 200]:
            dump = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "logs": logs
            }
            save_path = os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}_{case_num}.pt")
            torch.save(dump, save_path)
            tqdm.write(f"Saved checkpoint at {epoch+1}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--loss-scaling", type=str, default="average", choices=["average", "learnt_scalar"])
    parser.add_argument("--dynamic-seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--inspect", type=str, default="off")
    args = parser.parse_args()

    saved_std_filename = "./cc595k_all_384_models_std.npy"
    root_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_384"
    
    if os.path.exists(saved_std_filename):
        print("Loading stats from saved file.")
        std = np.load(saved_std_filename)
        std = torch.from_numpy(std)

        seeds = [i for i in range(5)]
        perturbations = get_multiseed_perturbations(seeds, std)

        print("With 1 (original) + 5 (perturbed).")
        main(args, case_num=6, perturbations=perturbations)
    
    else:
        std = get_onhand_std(root_folder=root_folder)
        embeddings = torch.from_numpy(np.load(os.path.join(root_folder, "vit_small_patch16_224", "embeddings.npy")))
        print(std.shape, embeddings.shape)

        seeds = [i for i in range(5)]
        perturbations = get_multiseed_perturbations(seeds, std)

        np.save(saved_std_filename, std)

        print("With 1 (original) + 5 (perturbed).")
        main(args, case_num=6, perturbations=perturbations)
