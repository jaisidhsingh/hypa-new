import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.datasets as torch_datasets


from src.models import MlpMapper, CustomVLM
from src.configs.model_configs import model_configs
from src.data.classification_datasets import ImageClassificationDataset
from perturb import *

"""
# TODO:
# 1. save imagenet embeddings (og+perturbed)
# 2. get hausdorf distance between the og and perturbed clusters
# 3. save class_prompt_embeddings across epochs (with the one from epoch 0)
# 4. plot
#   a. the class_prompt_embeddings and the og clusters across the epochs
#   b. the distance of the class_prompt_embeddings from the og cluster centroid across the epochs
#   c. the distance of the class_prompt_embeddings from the og cluster centroid as a fraction of the average hausdort distance 
"""

@torch.no_grad()
def get_class_name_embeddings(model, dataset):
    class_prompts = [f"a photo of a {c}" for c in dataset.classes]
    class_embeddings = model.encode_text(class_prompts)
    return class_embeddings


@torch.no_grad()
def get_classification_embeddings(args, model, loader, add_perturbation=False):
    num_images_per_class = len(loader.dataset) // len(loader.dataset.classes)
    embeddings = {
        "image": {i:torch.zeros(num_images_per_class, 384) for i in range(len(loader.dataset.classes))},
        "text": torch.zeros(len(loader.dataset.classes), 384)
    }
    
    class_embeddings = get_class_name_embeddings(model, loader.dataset)
    class_embeddings = class_embeddings.cpu().numpy()
    embeddings["text"] = class_embeddings

    image_embedding_store = []
    labels_store = []

    autocast = torch.cuda.amp.autocast

    bar = tqdm(total=len(loader))
    for idx, (images, labels) in tqdm(enumerate(loader)):
        images = images.float().to(args.device)
        labels = labels.long()
        with autocast():
            image_embeddings = model(images)
        
        batch_size = image_embeddings.shape[0]
        image_embedding_store.append(image_embeddings)
        labels_store.append(labels)
        bar.update(1)

    image_embedding_store = torch.cat(image_embedding_store, dim=0).cpu().numpy()
    labels_store = torch.cat(labels_store, dim=0).cpu().numpy()

    unique_labels = [i for i in range(len(loader.dataset.classes))]
    for lbl in unique_labels:
        embeddings["image"][lbl] = image_embedding_store[labels_store == lbl]
    

    if add_perturbation:
        stats = get_embedding_stats(image_embedding_store)
        seeds = [int(x) for x in args.seeds.split(",")]
        perturbations = get_perturbations_at_stats(stats, seeds)
        for k, v in embeddings["image"].items():
            embeddings["image"][k] = perturb_embeddings_statically(v, seeds, perturbations)

    print(f"""Created embedding store of {len(embeddings["image"].keys())} classes. Saved embeddings for each class have dim: {embeddings["image"][0].shape}""")
    return embeddings


def main(args):
    torch.manual_seed(args.random_seed)
    image_encoder = model_configs.ID_multi_mapper_configs[384][args.encoder_index]
    print(image_encoder)
    model = CustomVLM(image_encoder, args.text_encoder)
    model.mapper = MlpMapper(768, [], 384).to(args.device)

    if args.ckpt != "off":
        ckpt = torch.load(args.ckpt)["model"]
        model.mapper.load_state_dict(ckpt)
        print("load checkpoint")

    torch.compile(model)

    root_mapping = {
        "imagenet1k": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
        "cifar10": "/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
        "cifar100": "/home/mila/s/sparsha.mishra/scratch/cifar-100-python",
    }
    args.dataset = "cifar10"
    kwargs = {
        "feature_dataset": args.dataset,
        "root": root_mapping[args.dataset],
        "transform": model.image_encoder.transform
    }
    dataset = ImageClassificationDataset(kwargs)
    print(len(dataset))
    dataset = torch_datasets.CIFAR10(root=root_mapping["cifar10"], train=False, download=True, transform=model.image_encoder.transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    embeddings = get_classification_embeddings(args, model, loader, add_perturbation=args.add_perturbation)

    np.save(os.path.join(args.save_folder, "radius_analysis_embeddings.npy"), embeddings)
    print("All saved.")


def class_embeddings_across_epochs(args):
    torch.manual_seed(args.random_seed)
    image_encoder = model_configs.ID_multi_mapper_configs[384][args.encoder_index]
    model = CustomVLM(image_encoder, args.text_encoder)
    model.mapper = MlpMapper(768, [], 384).to(args.device)

    ckpt_epochs = [x for x in range(1, 11)] #[int(x) for x in args.ckpt_epochs]
    output = {}
    for epoch in ckpt_epochs:
        if args.ckpt != "off":
            path = "/".join(args.ckpt.split("/")[:-1]) + f"/ckpt_{epoch}.pt"
            print(path)
            ckpt = torch.load(args.ckpt)["model"]
            model.mapper.load_state_dict(ckpt)
            print("load checkpoint")

        torch.compile(model)

        root_mapping = {
            "imagenet1k": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
            "cifar10": "/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
            "cifar100": "/home/mila/s/sparsha.mishra/scratch/cifar-100-python",
        }
        kwargs = {
            "feature_dataset": "cifar10",
            "root": root_mapping["cifar10"],
            "transform": model.image_encoder.transform
        }
        dataset = ImageClassificationDataset(kwargs)
        print(len(dataset))
        class_embeddings = get_class_name_embeddings(model, dataset)
        print(class_embeddings.shape)
        output[epoch] = class_embeddings

    np.save(os.path.join(args.save_folder, "class_embeddings_over_epochs.npy"), output)
    print("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # global
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-wandb", type=bool, default=False)
    parser.add_argument("--add-perturbation", type=bool, default=False)
    parser.add_argument("--exp-name", type=str, default="uncond-static-1+5")
    parser.add_argument("--dataset", type=str, default="cifar10")
    # model
    parser.add_argument("--encoder-index", type=int, default=0)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--logit-scale", type=float, default=100.0)
    # paths
    parser.add_argument("--ckpt", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/theoretical/unconditional_checkpoints/static_vits_cc595k/seed_0/ckpt_1.pt")
    parser.add_argument("--save-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/theoretical")
    # training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    # seeding
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    args = parser.parse_args()
    main(args)
