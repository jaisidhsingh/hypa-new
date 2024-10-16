import os
import sys
import json
import argparse

import torch
from torch.utils.data import DataLoader

from src.evaluation import *
from src.models import MlpMapper, CustomVLM
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.data.image_caption_datasets import ImageCaptionDataset
from src.data.classification_datasets import ImageClassificationDataset


def load_separate_ckpt(args, model):
    path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/{args.exp_name}/seed_{args.seed}/ckpt_{args.epoch}.pt"
    ckpt = torch.load(path)["model"]
    model.mapper.load_state_dict(ckpt)
    model.eval()
    return model

def load_mm_ckpt(args, model):
    path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/{args.exp_name}/seed_{args.seed}/ckpt_{args.epoch}.pt"
    [weight, bias] = torch.load(path)["model"][args.encoder_index]
    model.mapper.layers[0].weight.data = weight
    model.mapper.layers[0].bias.data = bias
    model.eval()
    return model

def eval_retrieval(args, model, transform):
    config = data_configs.image_caption_dataset_configs["mscoco_val"]
    config.update({"transform": transform, "feature_dataset": "mscoco_val"})

    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False)
    recalls = coco_captions_eval(model, loader)
    return recalls


def eval_classification(args, model, transform):
    kwargs = {
        "feature_dataset": "imagenet",
        "root": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
        "transform": transform
    }
    dataset = ImageClassificationDataset(kwargs)
    loader = DataLoader(dataset, batch_size=1024, num_workers=4, pin_memory=True)
    accuracy = image_classification_eval(model, loader)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # main args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp-name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-type", type=str, default="sep")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--encoder-index", type=int, default=0)
    parser.add_argument("--benchmarks", type=str, default="mscoco,cifar10,cifar100,imagenet1k")
    parser.add_argument("--clip-version", type=str, default="off")
    # model args
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    # get args
    args = parser.parse_args()

    benchmark_mapping = {
        "mscoco": eval_retrieval,
        "cifar10": eval_classification,
        "cifar100": eval_classification,
        "imagenet1k": eval_classification,
    }
    [exp_name, run_type, epoch, encoder_index, benchmarks] = sys.argv[1:]

    benchmarks = benchmarks.split(",")
    metrics = {}
    result_save_file = "./eval_results.json"

    if args.clip_version == "off":
        image_encoder = model_configs.ID_multi_mapper_configs[args.image_embed_dim][args.encoder_index]
        model = CustomVLM(image_encoder, args.text_encoder)
        model.mapper = MlpMapper(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
        
        if args.run_type == "sep":
            model = load_separate_ckpt(args, model)
        elif args.run_type == "mm":
            model = load_mm_ckpt(args, model)
        
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, model.image_encoder.transform)
            metrics[bench] = metric

    else:
        model, preprocess = clip.load(args.clip_version, device=args.device)
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, preprocess)
            metrics[bench] = metric 
    
    result = {exp_name: metrics}

    with open(result_save_file, "r") as f:
        saved_results = json.load(f)

    saved_results.update(result)

    with open(result_save_file, "w") as f:
        json.dump(saved_results, f)
    
    print("All done and saved.")
    