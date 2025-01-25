import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.embedding_datasets import ImageEmbeddings, TextEmbeddings
from models.param_decoders import MLP
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from training.schedulers import cosine_lr


def main(args):
    encoder_names = model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"][:args.num_image_encoders]

    image_loaders, image_datasets = {}, {}
    loader_len = 0
    for name in enumerate(encoder_names):
        args.image_encoder = name
        image_datasets[name] = ImageEmbeddings(data_configs.separate_embedding_dataset_configs(args), split=None, args=None)
        image_loaders[name] = DataLoader(image_datasets[name], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        loader_len = len(image_loaders[name])
    
    for k, v in image_loaders.items():
        image_loaders[k] = iter(v)
    
    text_dataset = TextEmbeddings(data_configs.separate_embedding_dataset_configs(args), split=None, args=None)
    text_loader = DataLoader(text_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    model = MLP(args.text_embed_dim, [], args.text_embed_dim).to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_lr(optimizer, args.learning_rate, warmup_length=args.warmup_steps, steps=args.num_epochs * loader_len)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.amp.autocast

    bar = tqdm(total=range(args.num_epochs))
    logit_scale = torch.tensor(np.log(args.logit_scale)).to(args.device)

    for epoch in range(args.num_epochs):
        corrects = {name: 0 for name in encoder_names}
        total = {name: 0 for name in encoder_names}

        for idx, text_embeddings in enumerate(text_loader):
            text_embeddings = text_embeddings.float().to(args.device)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)



    

