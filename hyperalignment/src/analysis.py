import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import ImageEncoder
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from data.classification_datasets import ImageClassificationDataset


ROOT_MAPPING = {
    "imagenet1k": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
    "cifar10": "/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
    "cifar100": "/home/mila/s/sparsha.mishra/scratch/cifar-100-python",
}

@torch.no_grad()
def embed_imagenet(args):
    model = ImageEncoder(args.image_encoder).to(args.device)

    dataset_name = "imagenet1k"
    kwargs = {
        "feature_dataset": dataset_name,
        "root": ROOT_MAPPING[dataset_name],
        "transform": model.transform
    }
    dataset = ImageClassificationDataset(kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    autocast = torch.amp.autocast
    bar = tqdm(total=len(loader))
    store = {"inputs": [], "labels": []}

    for idx, (images, labels) in enumerate(loader):
        images = images.float().to(args.device)
        labels = labels.long()

        with autocast(args.device):
            image_features = model(images)
        
        store["inputs"].append(image_features)
        store["labels"].append(labels)

        bar.set_description(f"{args.image_encoder}")
        bar.update(1)
    
    bar.close()
    
    save_folder = os.path.join(args.results_folder, f"dim_{args.image_embed_dim}", args.image_encoder)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, "embedded_data.pt")
    torch.save(store, save_path)


def main(args):
    encoder_names = model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"]
    for name in encoder_names:
        args.image_encoder = name
        embed_imagenet(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--results-folder", type=str, default="/network/scratch/s/sparsha.mishra/hyperalignment/results/image_embeddings/icml/eval/imagenet1k")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    
    args = parser.parse_args()
    main(args)
