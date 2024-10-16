# TO DEBUG
import json
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
from src.data.image_caption_datasets import ImageCaptionDataset
from src.configs.model_configs import model_configs
from src.configs.data_configs import data_configs


def compute_retrieval(a2b_sims, return_ranks=False):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    # medr = np.floor(np.median(ranks)) + 1
    # meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10} #, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict


@torch.no_grad()
def coco_captions_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0

    if progress_bar:
        bar = tqdm(total=len(loader))
    logit_scale = torch.tensor(np.log(100.0)).to(device)

    image_store = []
    text_store = []
    D_img = 0

    for idx, (images, captions) in enumerate(loader):
        batch_size = images.shape[0]
        images = images.float().to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        D_img = image_features.shape[-1]
        # image_features = image_features.repeat((5, 1))
        image_store.append(image_features)

        # average the embeddings of the 5 captions per image
        caption_store = []
        for item in captions:
            item = item[:5]  #[item[0]]
            if using_clip:
                item = clip.tokenize(item).to(device)
            item_embeddings = model.encode_text(item)
            item_embeddings /= item_embeddings.norm(dim=-1, keepdim=True)
            caption_store.append(item_embeddings.unsqueeze(0))
        text_store.append(torch.cat(caption_store, dim=0))
        
        if progress_bar:
            bar.set_postfix({"batch_index": idx})
            bar.update(1)
        
    image_store = torch.cat(image_store, dim=0).view(5000, D_img)
    text_store = torch.cat(text_store, dim=0).view(5000, 5, D_img)

    mean_recalls = {"r1": 0, "r5": 0, "r10": 0}
    for i in range(1):
        sim = logit_scale * (image_store @ text_store[:, i, :].T)
        res = compute_retrieval(sim.cpu().numpy())
        for k in res.keys():
            mean_recalls[k] += res[k]
    
    # for k in mean_recalls.keys():
    #     mean_recalls[k] /= 5
    
    return mean_recalls


def coco_captions_eval2(model, loader, device="cuda"):
    correct, total = 0, 0

    logit_scale = torch.tensor(np.log(100.0)).to(device)

    image_store = []
    text_store = []
    D_img = 0

    for idx, (images, captions) in enumerate(loader):
        batch_size = images.shape[0]
        images = images.float().to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        D_img = image_features.shape[-1]

        captions = [item[0] for item in captions]
        text_features = model.encode_text(captions)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sim = logit_scale * (image_features @ text_features.T)
        preds = sim.argmax(dim=-1)
        labels = torch.arange(batch_size, dtype=torch.long).to(device)
        correct += (preds == labels).sum().item()
        total += batch_size
    
    return round(correct/total * 100, 2)

def eval_per_seed_and_epoch(args, random_seed, epoch, model, loader, ie_index):
    torch.manual_seed(random_seed)
    indices = [int(x) for x in args.ie_indices.split(",")]

    model.mapper = MlpMapper(768, [], 768).to(args.device)
    ckpt_path = os.path.join(args.checkpoint_folder, args.type, args.mode, args.name, f"seed_{random_seed}", f"ckpt_{epoch}.pt")

    if args.mode == "separate":
        ckpt = torch.load(ckpt_path)["model"]
        model.mapper.load_state_dict(ckpt)
    
    if args.mode == "joint":
        ckpt = torch.load(ckpt_path)["mapper_params"][indices.index(ie_index)]
        model.mapper.layers[0].weight.data = ckpt[0]
        model.mapper.layers[0].bias.data = ckpt[1]
    
    if args.mode == "multi-mapper":
        ckpt = torch.load(ckpt_path)["mapper_params"][indices.index(ie_index)]
        model.mapper.layers[0].weight.data = ckpt[0]
        model.mapper.layers[0].bias.data = ckpt[1]

    accuracy = coco_captions_eval(model, loader, progress_bar=args.progress_bar, using_clip=False, device=args.device)
    # print(ie_index, accuracy)
    return accuracy


def main(args):
    ie_indices = [int(x) for x in args.ie_indices.split(",")]

    # iterate over all models
    for ie_index in ie_indices:
        te_index = 0 #root.split("_")[1].split("-")[-1]
        
        if args.mode == "separate":
            args.name = f"id_vitr_ie-{ie_index}_te-0.cc3m595k_id_vitr_raw_epochs-40"
        elif args.mode == "joint":
            args.name = args.name + f"_ie-{ie_index}"

        image_encoder = model_configs.ID_experiment_configs[args.type]["image_encoders"][ie_index]
        text_encoder = model_configs.ID_experiment_configs[args.type]["text_encoders"][te_index]

        model = CustomVLM(image_encoder, text_encoder)

        config = data_configs.image_caption_dataset_configs["mscoco_val"]
        config.update({"transform": model.image_encoder.transform, "feature_dataset": "mscoco_val"})
        dataset = ImageCaptionDataset(config)
        loader = DataLoader(dataset, batch_size=1024, pin_memory=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn, shuffle=False)

        epochs = [int(x) for x in args.epochs.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        accuracies = np.zeros((len(epochs), len(seeds)))

        for i, epoch in enumerate(epochs):
            for j, random_seed in enumerate(seeds):
                accuracy = eval_per_seed_and_epoch(args, random_seed, epoch, model, loader, ie_index)
                accuracies[i][j] = accuracy
        
        avg = [round(x, 2) for x in accuracies.mean(axis=1).tolist()]
        std = [round(x, 2) for x in accuracies.std(axis=1).tolist()]

        # results_file_name = f"mscoco_{args.type}_{args.mode}.csv"
        # with open(os.path.join(args.results_folder, results_file_name), "w") as f:
        #     print("CSV file opened for logging")
        
        # results_path = os.path.join(args.results_folder, results_file_name)
        # df = pd.read_csv(results_path)

        # df has column names as follows:
        # image_encoder, text_encoder, train_dataset, epoch_1_r@1_mean, ..., epoch_10_r@1_mean, epoch_1_r@1_std, ..., epoch_10_r@1_std 
        new_row = {
            "image_encoder": image_encoder, 
            "text_encoder": text_encoder,
            "train_dataset": args.train_dataset
        }
        new_row.update({f"epoch_{ep}_r@1_mean": x for (ep, x) in zip(epochs, avg)})
        new_row.update({f"epoch_{ep}_r@1_std": x for (ep, x) in zip(epochs, std)})

        with open(os.path.join(args.results_folder, "mscoco_id_vitr.json"), "r") as f:
            logged_data = json.load(f)
        
        logged_data.update({args.name: new_row})
        
        with open(os.path.join(args.results_folder, "mscoco_id_vitr.json"), "w") as f:
            json.dump(logged_data, f)

        # df = df.append(new_row, ignore_index=True)
        # df.to_csv(results_path, index=False)

        print(f"{image_encoder}<-->{text_encoder} evaluated and saved.")
    
    print("All done!")


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="separate")
    parser.add_argument("--type", type=str, default="id_vitr")
    parser.add_argument("--train-dataset", type=str, default="cc3m595k")
    parser.add_argument("--ie-indices", type=str, default="0,1,3,4,5")
    parser.add_argument("--name", type=str, default="ie-0_te-0.cc3m595k_epochs-40")
    parser.add_argument("--epochs", type=str, default="1,5,10")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/projects/Hyper-Alignment/hyperalignment/results")
    parser.add_argument("--progress-bar", type=bool, default=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)
