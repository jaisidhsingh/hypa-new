import os
import math
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import *
from src.models import *
from src.training.loss_functions import ClipLoss
from src.configs.data_configs import data_configs


def main(args):
    dataset_config = data_configs.separate_embedding_dataset_configs(args)
    dataset = SeparateEmbeddings(dataset_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    embedding = torch.empty((1, args.cond_emb_dim))
    nn.init.normal_(embedding, mean=0, std=1/math.sqrt(args.cond_emb_dim))

    embedding = nn.Parameter(embedding)
    hypnet = HyperNetwork(args)

    optimizer = torch.optim.Adam([embedding], lr=args.learning_rate)
    criterion = ClipLoss()

    for epoch in range(args.num_epochs):
        correct, total = 0, 0
        running_loss = 0.0
        for idx, (image_embeddings, text_embeddings) in enumerate(loader):
            image_embeddings = image_embeddings.to(args.device)
            text_embeddings = text_embeddings.to(args.device)

            optimizer.zero_grad()
            output = hypnet.decode_conditional_embedding(image_embeddings, embedding)
            loss = criterion(output(text_embeddings), image_embeddings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += image_embeddings.size(0)
            correct += (output.argmax(dim=1) == text_embeddings.argmax(dim=1)).sum().item()
        
        running_loss /= idx+1
        accuracy = round(correct/total * 100, 2)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss}, Accuracy: {accuracy}%")