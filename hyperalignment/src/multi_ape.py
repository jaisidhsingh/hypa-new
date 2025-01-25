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


