import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")
import clip
import torch
from torch.utils.data import DataLoader

from models import CustomVLM, TextEncoder, ImageEncoder
from models.param_decoders import MLP
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from data.image_caption_datasets import ImageCaptionDataset
from data.classification_datasets import ImageClassificationDataset


def load_ape_ckpt(args, model):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints"
    path = os.path.join(folder, "APE_final", args.image_encoder, args.text_encoder, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    ckpt = torch.load(path)["model"]
    model.load_state_dict(ckpt)
    model = model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def emb_eval_classification(args, model, transform, dataset):
    imagenet_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/image_embeddings/icml/eval/imagenet1k"
    path = os.path.join(imagenet_folder, f"dim_{args.image_embed_dim}", args.image_encoder, "embedded_data.pt")
    data = torch.load(path)
    
    image_features = data["inputs"].to(args.device)
    labels = data["labels"].to(args.device)
    
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
    te = TextEncoder(args.text_encoder)
    logit_scale = torch.tensor(np.log(100.0)).to(args.device)
    class_prompt = [f"a photo of a {c}" for c in dataset.classes]
    class_features = te.encode_text(class_prompt).to(args.device)

    total = len(dataset)
    assert total == image_features.shape[0], "[ERROR]"
    mapped_features = model(class_features).to(args.device)
    mapped_features /= mapped_features.norm(dim=-1, keepdim=True)

    sim = logit_scale * (image_features @ mapped_features.T)
    corrects = (sim.argmax(dim=-1) == labels).sum().item()
    accuracy = round(corrects/total * 100, 2)
    return accuracy, 0


def ape_main(args):
    args.image_encoder = model_configs.ID_multi_mapper_configs[args.image_embed_dim][args.encoder_index]
    print(args.image_encoder, args.text_encoder, args.encoder_index)
    out = {"image_encoder": args.image_encoder, "seed": args.seed, "eval": {}}
    out["text_encoder"] = args.text_encoder
    
    for epoch in [1, 2, 5, 10, 20]:
        args.epoch = epoch
        benchmark_mapping = {
            "imagenet1k": emb_eval_classification,
        }

        benchmarks = ["imagenet1k"] 
        metrics = {}
        
        transform = ImageEncoder(args.image_encoder).transform
        model = MLP(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
        model = load_ape_ckpt(args, model)
        
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, transform, bench)[0]
            metrics[bench] = metric
        
        result = {f"epoch_{args.epoch}": metrics}
        out["eval"].update(result)
     
    return out


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
    parser.add_argument("--benchmarks", type=str, default="imagenet1k")
    parser.add_argument("--clip-version", type=str, default="off")
    # model args
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-encoders", type=int, default=6)
    parser.add_argument("--encoder-batch", type=int, default=6)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--image-encoder", type=str, default="vit_small_patch16_224")
    # get args
    args = parser.parse_args()

    args.exp_name = "deit3_large_patch16_384.fb_in22k_ft_in1k"
    # args.encoder_index = 0
    args.image_embed_dim = 1024
    args.text_embed_dim = 384
    args.text_encoder = "all-MiniLM-L12-v2"
    args.num_encoders = 1
    args.encoder_batch = 1
    args.encoder_index = 3

    res = {}
    for index in range(args.num_encoders):
        print(index)
        # args.encoder_index = index
        out = ape_main(args)
        out.update({"encoder_index": args.encoder_index})
        res[out["image_encoder"]] = out
    
    print(res)
