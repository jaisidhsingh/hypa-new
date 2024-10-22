import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader


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


def get_class_name_embeddings(model, dataset):
    class_prompts = [f"a photo of a {c}" for c in dataset.classes]
    class_embeddings = model.encode_text(class_prompts)
    return class_embeddings


def get_classification_embeddings(args, model, loader, add_perturbation=False):
    num_images_per_class = len(loader.dataset) / len(loader.dataset.classes)
    embeddings = {
        "image": {i:torch.zeros(num_images_per_class, args.image_embed_dim) for i in range(len(loader.dataset.classes))},
        "text": torch.zeros(len(loader.dataset.classes, args.image_embed_dim))
    }
    
    class_embeddings = get_class_name_embeddings(model, loader.dataset)
    class_embeddings = class_embeddings.cpu()
    embeddings["text"] = class_embeddings

    image_embedding_store = torch.zeros((len(loader.dataset), args.image_embed_dim))
    labels_store = torch.zeros(len(loader.dataset))
    previous = 0
    for idx, (images, labels) in enumerate(loader):
        images = images.float().to(args.device)
        image_embeddings = model.encode_image(images)
        batch_size = image_embeddings.shape[0]

        image_embedding_store[previous:previous+batch_size] = image_embeddings.cpu()
        labels_store[previous:previous+batch_size] = labels.cpu()
    
    unique_labels = [i for i in range(len(loader.dataset.classes))]
    for lbl in unique_labels:
        embeddings["image"][lbl] = image_embedding_store[labels_store == lbl]
    
    stats = get_embedding_stats(image_embedding_store)

    if add_perturbation:
        seeds = [int(x) for x in args.seeds.split(",")]
        perturbations = get_perturbations_at_stats(stats, seeds)
        for k, v in embeddings["image"].items():
            embeddings["image"][k] = perturb_embeddings_statically(v, seeds, perturbations)

    print(f"""Created embedding store of {len(embeddings["image"].keys())} classes. Saved embeddings for each class have dim: {embeddings["image"][0].shape}""")
    return embeddings


def main(args):
    torch.manual_seed(args.random_seed)
    image_encoder = model_configs.ID_multi_mapper_configs[384][args.encoder_index]
    model = CustomVLM(image_encoder, args.text_encoder)
    model.mapper = MlpMapper(768, [], 384).to(args.device)

    if args.ckpt != "off":
        ckpt = torch.load(args.ckpt_path)["model"]
        model.mapper.load_state_dict(ckpt)
    
    root_mapping = {
        "imagenet1k": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
        "cifar10": "/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
        "cifar100": "/home/mila/s/sparsha.mishra/scratch/cifar-100-python",
    }
    kwargs = {
        "feature_dataset": dataset,
        "root": root_mapping[dataset],
        "transform": model.image_encoder.transform
    }
    dataset = ImageClassificationDataset(kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    embeddings = get_classification_embeddings(args, model, loader, add_perturbation=args.add_perturbation)

    np.save(os.path.join(args.save_folder, "radius_analysis_embeddings.npy"), embeddings)
    print("All saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # global
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-wandb", type=bool, default=False)
    parser.add_argument("--add-perturbation", type=bool, default=False)
    parser.add_argument("--exp-name", type=str, default="uncond-static-1+5")
    parser.add_argument("--dataset", type=str, default="imagenet1k")
    # model
    parser.add_argument("--encoder-index", type=int, default=0)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--logit-scale", type=float, default=100.0)
    # paths
    parser.add_argument("--save-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/theoretical")
    # training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    # seeding
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    args = parser.parse_args()
    main(args)
