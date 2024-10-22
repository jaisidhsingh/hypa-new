import os
import torch
import numpy as np

from src.models import MlpMapper, CustomVLM
from src.configs.model_configs import model_configs
from src.data.classification_datasets import ImageClassificationDataset

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


def get_classification_embeddings(args, model, loader, save_folder, add_perturbation=False):
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
    
    if add_perturbation:
        pass

    np.save(os.path.join(save_folder, "radius_analysis_embeddings.npy"), embeddings)
    print("All saved.")
