import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import ImageEncoder, TextEncoder
from configs.model_configs import model_configs
from configs.data_configs import data_configs
from data.classification_datasets import ImageClassificationDataset
from data.image_caption_datasets import ImageCaptionDataset


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

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
        
        store["inputs"].append(image_features.cpu())
        store["labels"].append(labels.cpu())

        bar.set_description(f"{args.image_encoder}")
        bar.update(1)
    
    bar.close()

    store["inputs"] = torch.cat(store["inputs"], dim=0)
    store["labels"] = torch.cat(store["labels"], dim=0)
    
    save_folder = os.path.join(args.results_folder, f"dim_{args.image_embed_dim}", args.image_encoder)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, "embedded_data.pt")
    torch.save(store, save_path)


@torch.no_grad()
def embed_coco_images(args):
    model = ImageEncoder(args.image_encoder).to(args.device)

    config = data_configs.image_caption_dataset_configs["mscoco_val"]
    config.update({"transform": ImageEncoder.transform, "feature_dataset": "mscoco_val"})

    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn, shuffle=False)

    autocast = torch.amp.autocast
    bar = tqdm(total=len(loader))
    store = {"image_features": []}

    for idx, (images, _) in enumerate(loader):
        images = images.float().to(args.device)

        with autocast(args.device):
            image_features = model(images)
        
        store["image_features"].append(image_features.cpu())
        bar.set_description(f"{args.image_encoder}")
        bar.update(1)
    
    bar.close()

    store["image_features"] = torch.cat(store["image_features"], dim=0)
    
    save_folder = os.path.join(args.results_folder2, f"dim_{args.image_embed_dim}", args.image_encoder)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, "embedded_data.pt")
    torch.save(store, save_path)


@torch.no_grad()
def embed_coco_captions(args):
    model = TextEncoder(args.text_encoder)

    config = data_configs.image_caption_dataset_configs["mscoco_val"]
    config.update({"transform": None, "feature_dataset": "mscoco_val"})

    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn, shuffle=False)

    autocast = torch.amp.autocast
    bar = tqdm(total=len(loader))
    store = {"text_features": []}

    for idx, (_, captions) in enumerate(loader):
        bs = len(captions)
        with autocast(args.device):
            stack = []
            for j in range(5):
                inputs = [item[j] for item in captions]
                text_features = model.encode_text(inputs).unsqueeze(1)
                stack.append(text_features)
            
        store["text_features"].append(torch.cat(stack, dim=1).view(bs, 5, args.text_embed_dim))
        bar.set_description(f"{args.text_encoder}")
        bar.update(1)
    
    bar.close()

    store["text_features"] = torch.cat(store["text_features"], dim=0).view(len(dataset), 5, args.text_embed_dim)
    
    save_folder = os.path.join(args.results_folder3, f"dim_{args.text_embed_dim}", args.text_encoder)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, "embedded_data.pt")
    torch.save(store, save_path)


def main_imagenet(args):
    encoder_names = model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"][args.offset:args.end]
    for name in encoder_names:
        print("Embedding imagenet for", name)
        args.image_encoder = name
        embed_imagenet(args)

def main_coco(args):
    # encoder_names = model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"][args.offset:args.end]
    # for name in encoder_names:
    #     print("Embedding imagenet for", name)
    #     args.image_encoder = name
    embed_coco_captions(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--results-folder", type=str, default="/network/scratch/s/sparsha.mishra/hyperalignment/results/image_embeddings/icml/eval/imagenet1k")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)

    parser.add_argument("--text-embed-dim", type=int, default=384)
    parser.add_argument("--text-encoder", type=str, default="all-MiniLM-L12-v2")
    parser.add_argument("--results-folder2", type=str, default="/network/scratch/s/sparsha.mishra/hyperalignment/results/image_embeddings/icml/eval/mscoco")
    parser.add_argument("--results-folder3", type=str, default="/network/scratch/s/sparsha.mishra/hyperalignment/results/text_embeddings/icml/eval/mscoco")
    
    args = parser.parse_args()
    main_coco(args)
