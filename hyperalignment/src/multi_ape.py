import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.embedding_datasets import SeparateEmbeddings
from models.param_decoders import MLP
from configs.data_configs import data_configs
from configs.model_configs import model_configs


def main(args):
    encoder_names = model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"][:args.num_image_encoders]

    loaders, datasets = {}, {}
    for name in enumerate(encoder_names):
        args.image_encoder = name
        datasets[name] = SeparateEmbeddings(data_configs.separate_embedding_dataset_configs(args))
        loaders[name] = DataLoader(datasets[name], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    for k, v in loaders.items():
        loaders[k] = iter(v)
    
    model = MLP(args.text_embed_dim, [], args.text_embed_dim).cuda()
    

