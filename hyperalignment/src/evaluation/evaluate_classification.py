# TO DEBUG

import os
import clip
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets as torch_datasets

from src.models import CustomVLM, MlpMapper
from src.configs.model_configs import model_configs
from src.configs.data_configs import data_configs


@torch.no_grad()
def image_classification_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0

    if progress_bar:
        bar = tqdm(total=len(loader))

    logit_scale = torch.tensor(np.log(100.0)).to(device)
    class_prompt = [f"a photo of a {c}" for c in loader.dataset.classes]
    
    if using_clip:
        class_prompt = clip.tokenize(class_prompt)
    class_features = model.encode_text(class_prompt).to(device)

    for idx, (images, labels) in enumerate(loader):
        batch_size = images.shape[0]

        if using_clip:
            captions = clip.tokenize(captions).to(device)

        images = images.float().to(device)
        image_features = model.encode_image(images)

        sim = logit_scale.exp() * image_features @ class_features.T
        preds = torch.argmax(sim, dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

        accuracy = round(correct/total * 100, 2)

        if progress_bar:
            bar.set_postfix({"accuracy": accuracy})

        bar.update(1)
    
    return accuracy


def eval_per_seed_and_epoch(args, random_seed, epoch, model, loader, ie_index):
    torch.manual_seed(random_seed)

    model.mapper = MlpMapper(768, [], 768).to(args.device)
    ckpt_path = os.path.join(args.checkpoint_folder, args.type, args.mode, args.name, f"seed_{random_seed}", f"ckpt_{epoch}.pt")

    if args.mode == "separate":
        ckpt = torch.load(ckpt_path)["model"]
        model.mapper.load_state_dict(ckpt)
    
    if args.mode == "joint":
        ckpt = torch.load(ckpt_path)["mapper_params"][ie_index]
        model.mapper.layers[0].weight.data = ckpt[0]
        model.mapper.layers[0].bias.data = ckpt[1]
    
    if args.mode == "multi-mapper":
        ckpt = torch.load(ckpt_path)["mapper_params"][ie_index]
        model.mapper.layers[0].weight.data = ckpt[0]
        model.mapper.layers[0].bias.data = ckpt[1]

    accuracy = image_classification_eval(model, loader, progress_bar=False, using_clip=False, device=args.device)
    return accuracy


def main(args):
    ie_indices = [int(x) for x in args.ie_indices.split(",")]

    # iterate over all models
    for ie_index in ie_indices:
        te_index = 0 #root.split("_")[1].split("-")[-1]
        
        if args.mode == "separate":
            args.name = f"ie-{args.ie_index}_te-0.cc3m595k_epochs-40"

        image_encoder = model_configs.ID_experiment_configs[args.type]["image_encoders"][ie_index]
        text_encoder = model_configs.ID_experiment_configs[args.type]["text_encoders"][te_index]

        model = CustomVLM(image_encoder, text_encoder)

        config = data_configs.image_caption_dataset_configs["mscoco_val"]
        config.update({"transform": model.image_encoder.transform})
        dataset = torch_datasets.CocoCaptions(**config)
        loader = DataLoader(dataset, batch_size=1024, pin_memory=True, collate_fn=dataset.collate_fn, shuffle=False)

        epochs = [int(x) for x in args.epochs.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        accuracies = np.zeros((epochs, seeds))

        for i, epoch in enumerate(epochs):
            for j, random_seed in enumerate(seeds):
                accuracy = eval_per_seed_and_epoch(args, random_seed, epoch, model, loader, ie_index)
                accuracies[i][j] = accuracy
        
        avg = [round(x, 2) for x in accuracies.mean(axis=1).tolist()]
        std = [round(x, 2) for x in accuracies.std(axis=1).tolist()]

        results_file_name = f"{args.eval_dataset}_{args.type}_{args.mode}.csv"
        results_path = os.path.join(args.results_folder, results_file_name)
        df = pd.read_csv(results_path)

        # df has column names as follows:
        # image_encoder, text_encoder, train_dataset, epoch_1_r@1_mean, ..., epoch_10_r@1_mean, epoch_1_r@1_std, ..., epoch_10_r@1_std 
        new_row = {
            "image_encoder": image_encoder, 
            "text_encoder": text_encoder,
            "train_dataset": args.train_dataset
        }
        new_row.update({f"epoch_{ep}_r@1_mean": x for (ep, x) in zip(epochs, avg)})
        new_row.update({f"epoch_{ep}_r@1_std": x for (ep, x) in zip(epochs, std)})

        df = df.append(new_row, ignore_index=True)
        df.to_csv(results_path, index=False)

        print(f"{image_encoder}<-->{text_encoder} evaluated and saved.")
    
    print("All done!")


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="separate")
    parser.add_argument("--type", type=str, default="id_vitr")
    parser.add_argument("--train-dataset", type=str, default="cc3m595k")
    parser.add_argument("--eval-dataset", type=str, default="imagenet")
    parser.add_argument("--ie-indices", type=str, default="0,1,3,4,5")
    parser.add_argument("--name", type=str, default="ie-0_te-0.cc3m595k_epochs-40")
    parser.add_argument("--epochs", type=str, default="1,5,10")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
    parser.add_argument("--results-folder", type=str, default="results")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)

