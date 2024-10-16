# TO DEBUG
import sys
import os
import clip
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets as torch_datasets

from src.models import CustomVLM, MlpMapper
from src.configs.model_configs import model_configs
from src.configs.data_configs import data_configs
from src.data.image_caption_datasets import ImageCaptionDataset
from src.data.classification_datasets import ImageClassificationDataset
warnings.simplefilter("ignore")

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
    for i in range(5):
        sim = logit_scale * (image_store @ text_store[:, i, :].T)
        res = compute_retrieval(sim.cpu().numpy())
        for k in res.keys():
            mean_recalls[k] += res[k]
    
    for k in mean_recalls.keys():
        mean_recalls[k] /= 5
    
    return mean_recalls


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
    class_features = F.normalize(class_features, dim=-1)

    for idx, (images, labels) in enumerate(loader):
        batch_size = images.shape[0]
        labels = labels.long().to(device)

        if using_clip:
            captions = clip.tokenize(captions).to(device)

        images = images.float().to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        sim = logit_scale.exp() * image_features @ class_features.T
        preds = torch.argmax(sim, dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

        accuracy = round(correct/total * 100, 2)

        if progress_bar:
            bar.set_postfix({"accuracy": accuracy})
            bar.update(1)
    
    return accuracy


if __name__ == "__main__":
    device = "cuda"
    case_num = int(sys.argv[1])
    ep = int(sys.argv[2])

    model = CustomVLM(image_encoder_name="vit_small_patch16_224", text_encoder_name="sentence-t5-base")
    model.mapper = MlpMapper(768, [], 384).to(device)

    ckpt_save_folder = "/home/mila/s/sparsha.mishra/scratch/hypa/checkpoints/uncond/vit_s16_with_5"
    ckpt = torch.load(os.path.join(ckpt_save_folder, f"ckpt_{ep}_{case_num}.pt"))["model"]

    exp_name = "norm"
    path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/{exp_name}/seed_0/ckpt_{ep}.pt"

    model.mapper.load_state_dict(ckpt)
    model.mapper.eval()

    # dataset = torch_datasets.CIFAR10(
    #     root="/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
    #     train=False,
    #     download=False,
    #     transform=model.image_encoder.transform
    # )
    # loader = DataLoader(dataset, batch_size=1024, num_workers=1, pin_memory=True)
    # accuracy = image_classification_eval(model, loader)
    # print(accuracy)

    # config = data_configs.image_caption_dataset_configs["mscoco_val"]
    # config.update({"transform": model.image_encoder.transform, "feature_dataset": "mscoco_val"})

    # dataset = ImageCaptionDataset(config)
    # loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False)
    # r_at_k = coco_captions_eval(model, loader)
    # print(r_at_k)

    kwargs = {
        "feature_dataset": "imagenet",
        "root": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
        "transform": model.image_encoder.transform
    }
    dataset = ImageClassificationDataset(kwargs)
    loader = DataLoader(dataset, batch_size=1024, num_workers=1, pin_memory=True)
    accuracy = image_classification_eval(model, loader)
    print(accuracy)
