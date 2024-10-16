import numpy as np
import torch
import os
import timm
import torch.nn as nn
import argparse
from tqdm import tqdm
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import argparse
import warnings
warnings.simplefilter("ignore")

from configs.model_configs import model_configs
from configs.data_configs import data_configs
# from models import TextEncoder, ImageEncoder
from data import ImageCaptionDataset


class TinEncoder(nn.Module):
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device

        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**self.config, is_training=False)
    
    def forward(self, x):
        x = self.model.forward_features(x)
        return x[:, 0, :]


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
        self.config = timm.data.resolve_model_data_config(self.backbone)
        self.transform = timm.data.create_transform(**self.config, is_training=False)

        self.act = nn.GELU()
        self.num_classes = num_classes
        self.fc = nn.Linear(384, 384)
        self.out = nn.Linear(384, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        features = self.fc(x)
        return features


@torch.no_grad()
def embed_images(args):
    save_folder_name = f"mscoco_train_tin" 
    save_folder = os.path.join(args.results_folder, "image_embeddings", "tin", save_folder_name, f"dim_{args.image_embed_dim}")
    os.makedirs(save_folder, exist_ok=True)

    autocast = torch.cuda.amp.autocast
    store = {}
    store[args.image_embed_dim] = {}

    for encoder_name in ["vit_small_patch16_224"]:
        folder_for_encoder = os.path.join(save_folder, encoder_name)
        os.makedirs(folder_for_encoder, exist_ok=True)

        image_encoder = Model(200).to(args.device) #TinEncoder(model_name=encoder_name, device=args.device)
        ckpt_path = os.path.join(args.checkpoint_folder, "tinyimagenet", f"seed_{args.seed}", "epoch30.pt")
        ckpt = torch.load(ckpt_path)["model_state_dict"]
        for k in ckpt.keys():
            print(k)
        # ckpt.pop("head.weight")
        # ckpt.pop("head.bias")
        image_encoder.backbone.load_state_dict(ckpt)
        
        torch.compile(image_encoder)

        kwargs = {"feature_dataset": "mscoco_train", "transform": image_encoder.transform}
        dataset_config = data_configs.image_caption_dataset_configs[kwargs["feature_dataset"]]
        kwargs.update(dataset_config)

        dataset = ImageCaptionDataset(kwargs)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)

        bar = tqdm(total=len(loader))
        features = np.zeros((len(dataset), args.image_embed_dim))

        for i, (images, captions) in enumerate(loader):
            del captions
            batch_size = images.shape[0]
            images = images.float().to(args.device)
            
            with autocast():
                image_features = image_encoder.forward_features(images)
            
            features[i:i+batch_size, :] = image_features.cpu()

            bar.update(1)
            bar.set_postfix({"encoder": encoder_name, "dim": args.image_embed_dim})

        bar.close()
        np.save(os.path.join(folder_for_encoder, f"embeddings_{args.seed}.npy"), features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", default="/home/mila/s/sparsha.mishra/scratch/hypa/results/image_embeddings")
    parser.add_argument("--checkpoint-folder", default="/home/mila/s/sparsha.mishra/scratch/hypa/checkpoints")
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    embed_images(args)
