import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from types import SimpleNamespace

from models import CustomVLM, ImageEncoder, TextEncoder
from data.image_caption_datasets import ImageCaptionDataset
from configs.model_configs import model_configs
from configs.data_configs import data_configs


@torch.no_grad()
def one_encoder_embeds_images(args):
    config = data_configs.image_caption_dataset_configs["cc3m595k"]
    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, shuffle=False)
    model = ImageEncoder(args.image_encoder)
    model = model.to(args.device)

    store = np.zeros((len(dataset), args.image_embed_dim), dtype=np.float32)

    for idx, (images, _) in tqdm(enumerate(loader)):
        bs = len(images)
        images = images.float().to(args.device)

        image_features = model(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()

        start = idx * args.batch_size
        end = start + bs 
        store[start : end] = image_features.astype(np.float32)
    
    save_folder = f"{args.image_results_folder}/dim_{args.image_embed_dim}/{args.image_encoder}"
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "cc3m595k_embeddings.npy"), store)
    print("Done for images")
        

@torch.no_grad()
def one_encoder_embeds_texts(args):
    config = data_configs.image_caption_dataset_configs["cc3m595k"]
    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, shuffle=False)
    model = TextEncoder(args.text_encoder)
    model = model.to(args.device)

    store = np.zeros((len(dataset), args.text_embed_dim), dtype=np.float32)

    for idx, (_, captions) in tqdm(enumerate(loader)):
        bs = len(captions)

        text_features = model(captions)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

        start = idx * args.batch_size
        end = start + bs 
        store[start : end] = text_features.astype(np.float32)
    
    save_folder = f"{args.text_results_folder}/dim_{args.text_embed_dim}/{args.text_encoder}"
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "cc3m595k_embeddings.npy"), store)
    print("Done for texts")


def main():
    args = SimpleNamespace(**{})
    args.device = "cuda"
    args.batch_size = 2048
    args.num_workers = 4

    args.image_embed_dim = 384
    args.image_encoder = "vit_small_patch16_224"
    args.image_results_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/image_embeddings/icml"
    one_encoder_embeds_images(args)
    
    args.text_embed_dim = 384
    args.text_encoder = "all-MiniLM-L12-v2"
    args.text_results_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/text_embeddings/icml"
    one_encoder_embeds_texts(args)


# def embed_images_per_family(config):
#     image_encoders = model_configs.ID_experiment_configs["multi_mapper"][config.image_embed_dim]["image_encoders"]
#     args = SimpleNamespace(**{})
#     args.device = "cuda"
#     args.image_embed_dim = config.image_embed_dim
    
#     for ie in image_encoders: