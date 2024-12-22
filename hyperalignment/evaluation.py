import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

import clip
import torch
from torch.utils.data import DataLoader

from src.models import MlpMapper, CustomVLM
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.data.image_caption_datasets import ImageCaptionDataset
from src.data.classification_datasets import ImageClassificationDataset


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

    report_dict = {"R@1": r1, "R@5": r5, "R@10": r10} 

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

    mean_recalls = {"R@1": 0, "R@5": 0, "R@10": 0}
    for i in range(1):
        sim = logit_scale * (image_store @ text_store[:, i, :].T)
        res = compute_retrieval(sim.cpu().numpy())
        for k in res.keys():
            mean_recalls[k] += res[k]
    
    return mean_recalls


@torch.no_grad()
def image_classification_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0

    if progress_bar:
        bar = tqdm(total=len(loader))

    logit_scale = torch.tensor(np.log(100.0)).to(device)
    class_prompt = [f"a photo of a {c}" for c in loader.dataset.classes]
    
    if using_clip:
        class_prompt = clip.tokenize(class_prompt).to(device)

    class_features = model.encode_text(class_prompt).to(device)

    for idx, (images, labels) in enumerate(loader):
        batch_size = images.shape[0]
        labels = labels.long().to(device)

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


def load_separate_ckpt(args, model):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper"
    path = os.path.join(folder, "separate", args.image_encoder, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    ckpt = torch.load(path)["model"]
    model.mapper.load_state_dict(ckpt)
    model.mapper = model.mapper.to(args.device)
    model.mapper.eval()
    return model


def load_ood_ckpt(args, model):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper"
    path = os.path.join(folder, args.ood_results_path)
    [weight, bias] = torch.load(path)["mapper_params"]
    model.mapper.layers[0].weight.data = weight.to(args.device)
    model.mapper.layers[0].bias.data = bias.to(args.device)
    model.mapper = model.mapper.to(args.device)
    model.mapper.eval()
    return model


def load_mm_ckpt(args, model):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper"
    path = os.path.join(folder, args.exp_name, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    [weight, bias] = torch.load(path)["mapper_params"][args.encoder_index]
    model.mapper.layers[0].weight.data = weight.to(args.device)
    model.mapper.layers[0].bias.data = bias.to(args.device)
    model.mapper = model.mapper.to(args.device)
    model.mapper.eval()
    return model


def eval_retrieval(args, model, transform):
    config = data_configs.image_caption_dataset_configs["mscoco_val"]
    config.update({"transform": transform, "feature_dataset": "mscoco_val"})

    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False)
    using_clip = args.clip_version != "off"
    recalls = coco_captions_eval(model, loader, using_clip=using_clip, device=args.device)
    return recalls


def eval_classification(args, model, transform, dataset):
    root_mapping = {
        "imagenet1k": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
        "cifar10": "/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
        "cifar100": "/home/mila/s/sparsha.mishra/scratch/cifar-100-python",
    }
    kwargs = {
        "feature_dataset": dataset,
        "root": root_mapping[dataset],
        "transform": transform
    }
    dataset = ImageClassificationDataset(kwargs)
    print(dataset.classes[0])
    loader = DataLoader(dataset, batch_size=1024, num_workers=4, pin_memory=True)
    using_clip = args.clip_version != "off"
    accuracy = image_classification_eval(model, loader, using_clip=using_clip, device=args.device)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # main args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp-name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-type", type=str, default="sep", choices=["sep", "mm", "ood"])
    parser.add_argument("--ood-results-path", type=str, default="ood_attempt_1.pt")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--encoder-index", type=int, default=0)
    parser.add_argument("--benchmarks", type=str, default="mscoco,cifar10,cifar100,imagenet1k")
    parser.add_argument("--clip-version", type=str, default="off")
    # model args
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--image-encoder", type=str, default="set-later")
    # get args
    args = parser.parse_args()

    benchmark_mapping = {
        "mscoco": eval_retrieval,
        "cifar10": eval_classification,
        "cifar100": eval_classification,
        "imagenet1k": eval_classification,
    }

    benchmarks = args.benchmarks.split(",")
    metrics = {}
    result_save_file = "./eval_results.json"

    if args.clip_version == "off":
        args.image_encoder = model_configs.ID_multi_mapper_configs[args.image_embed_dim][args.encoder_index]
        model = CustomVLM(args.image_encoder, args.text_encoder)
        model.mapper = MlpMapper(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
        
        if args.run_type == "sep":
            model = load_separate_ckpt(args, model)
        elif args.run_type == "mm":
            model = load_mm_ckpt(args, model)
        elif args.run_type == "ood":
            model = load_ood_ckpt(args, model)
        
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, model.image_encoder.transform, bench)
            metrics[bench] = metric

    else:
        model, preprocess = clip.load(args.clip_version, device=args.device)
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, preprocess, bench)
            metrics[bench] = metric 
    
    metrics.update({"config": vars(args)})
    result = {args.exp_name: metrics}

    with open(result_save_file, "r") as f:
        saved_results = json.load(f)

    saved_results.update(result)

    with open(result_save_file, "w+") as f:
        json.dump(saved_results, f)
    
    print("All done and saved.")
    
