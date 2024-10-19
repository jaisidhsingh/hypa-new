import torch
import numpy as np

from src.models import MlpMapper, CustomVLM
from src.configs.model_configs import model_configs
from src.data.classification_datasets import ImageClassificationDataset


def get_class_name_embeddings(model, dataset):
    class_prompts = [f"a photo of a {c}" for c in dataset.classes]
    class_embeddings = model.encode_text(class_prompts)
    return class_embeddings


def get_classification_embeddings(args, model, loader):
    num_images_per_class = len(loader.dataset) / len(loader.dataset.classes)
    embeddings = {
        "image": {i:torch.zeros(num_images_per_class, args.image_embed_dim) for i in range(len(loader.dataset.classes))},
        "text": torch.zeros(len(loader.dataset.classes, args.image_embed_dim))
    }
    # do something to update `embeddings` ...
    return embeddings
