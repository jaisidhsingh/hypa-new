import torch
import numpy as np
from torch.utils.data import DataLoader 

from models import CustomVLM, ImageEncoder, TextEncoder
from data.image_caption_datasets import ImageCaptionDataset
from configs.model_configs import model_configs
from configs.data_configs import data_configs


@torch.no_grad()
def encode_images(args):
    config = data_configs.image_caption_dataset_configs["mscoco_train"]
    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, shuffle=False)
    model = ImageEncoder(args.image_encoder)
    model = model.to(args.device)

    store = np.zeros((len(dataset), args.image_embed_dim), dtype=np.float32)

    for idx, (images, _) in enumerate(loader):
        images = images.to(args.device)
        image_features = model(images)
        store[idx * args.batch_size : (idx + 1) * args.batch_size] = image_features.cpu().numpy()
    
    np.save(f"{data_configs.STORE}/hyperalignment/results/image_embeddings/{args.feature_dataset}/dim_{args.image_embed_dim}.npy", store)
        

