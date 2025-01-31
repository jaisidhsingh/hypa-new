from PIL import Image
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


def load_separate_ckpt(args, model, vlm=False):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints"
    path = os.path.join(folder, "APE_final", args.image_encoder, args.text_encoder, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    ckpt = torch.load(path)["model"]
    if not vlm:
        model.load_state_dict(ckpt)
        model = model.to(args.device)
        model.eval()
    if vlm:
        model.mapper.load_state_dict(ckpt)
        model.mapper = model.mapper.to(args.device)
        model.mapper.eval()
    return model


def load_mm_ckpt(args, model, vlm=False):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper"   # multi_mapper
    path = os.path.join(folder, args.exp_name, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    chunk_size = args.num_encoders // 3 #int(args.exp_name.split("_")[1]) // 3
    
    if args.image_embed_dim == 384:
        offset = 0
    elif args.image_embed_dim == 768:
        offset = 1
    else:
        offset = 2

    index = int(offset * chunk_size) + args.encoder_index
    [weight, bias] = torch.load(path)["mapper_params"][index]
    # print(weight.shape, bias.shape)
    if not vlm:
        model.layers[0].weight.data = weight.to(args.device)
        model.layers[0].bias.data = bias.to(args.device)
        model = model.to(args.device)
        model.eval()
    if vlm:
        model.mapper.layers[0].weight.data = weight.to(args.device)
        model.mapper.layers[0].bias.data = bias.to(args.device)
        model.mapper = model.mapper.to(args.device)
        model.mapper.eval()
    
    return model

@torch.no_grad()
def main(args):
    prompts = {
        "dog": "High-contrast black and white photo of dog emphasizing dramatic lighting.",
        "cat": "cat image with a pleasing color palette using analogous colors that sit next to each other on the color wheel.",
        "house": "Produce an image of house with realistic details and lighting.",
        "car": "Depict car in a photo with a high degree of visual realism."
    }
    names = {
        "dog": "3_", "cat": "0_", "house": "0_", "car": "0_"
    }
    
    args.image_encoder = model_configs.ID_multi_mapper_configs[args.image_embed_dim][args.encoder_index]
    model = CustomVLM(args.image_encoder, args.text_encoder)
    model.mapper = MLP(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
    
    if args.run_type == "sep":
        model = load_separate_ckpt(args, model, vlm=True)
    elif args.run_type == "mm":
        model = load_mm_ckpt(args, model, vlm=True)
    
    prompt_features = model.encode_text([prompts[args.class_name]])
    prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
    image_feature_store = []
    for i in range(5):
        path = os.path.join("/home/mila/s/sparsha.mishra/projects/diff", args.class_name, names[args.class_name] + str(i) + ".png")
        image = Image.open(path).convert("RGB")
        image = model.image_encoder.transform(image).unsqueeze(0).to(args.device)
        image_features = model.encode_image(image)
        image_feature_store.append(image_features)
    
    image_feature_store = torch.stack(image_feature_store).view(5, args.image_embed_dim)
    image_feature_store /= image_feature_store.norm(dim=-1)
    sim = 100 * (image_feature_store @ prompt_features.T)
    sim = sim.cpu().view(5,).numpy().tolist()
    f = [(j, item) for j, item in enumerate(sim)]
    f.sort(key=lambda x: x[1])
    print(args.class_name)
    print(f)


    
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-encoders", type=int, default=6)
    parser.add_argument("--encoder-batch", type=int, default=6)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--image-encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--class-name", type=str, default="dog")
    # get args
    args = parser.parse_args()

    args.exp_name = "hnet_30-10_fmlp_c-32_bs-512_lr-1e-2"
    args.encoder_index = 6
    args.image_encoder = model_configs.ID_multi_mapper_configs[args.image_embed_dim][args.encoder_index]
    args.text_embed_dim = 768
    args.text_encoder = "sentence-t5-base"
    args.num_encoders = 30
    args.encoder_batch = 10
    main(args)
 