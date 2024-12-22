import os
import sys
import math
import argparse
import warnings
from tqdm import tqdm
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import *
from src.models import *
from src.training.loss_functions import ClipLoss
from src.configs.data_configs import data_configs

def main(args):
    torch.manual_seed(args.random_seed)

    param_shapes = [[args.largest_image_dim, args.largest_text_dim], [args.largest_image_dim]]
    image_embed_dims = [int(x) for x in args.image_embed_dims.split(",")]
    hidden_layer_factors = [int(x) for x in args.hidden_layer_factors.split(",")]

    embedding = nn.Embedding(1, args.hnet_cond_emb_dim)
    # nn.init.normal_(embedding, mean=0, std=1/math.sqrt(args.hnet_cond_emb_dim))
    # embedding.requires_grad = True
    for param in embedding.parameters():
        param.requires_grad = True
    embedding = embedding.to(args.device)
    print("Initialized embedding to auto-decode.")

    hnet = MultiMapperHypernet(
        param_shapes, cond_emb_dim=args.hnet_cond_emb_dim,
        num_cond_embs=args.hnet_ckpt_num_ie, image_embed_dims=image_embed_dims,
        hidden_layer_factors=hidden_layer_factors, rescale_factor=0.0
    )

    ckpt_path = os.path.join(args.hnet_ckpt_folder, args.hnet_ckpt_name, f"seed_{args.random_seed}", f"ckpt_{args.hnet_ckpt_epoch}.pt")
    hnet.load_state_dict(torch.load(ckpt_path)["model"])
    hnet = hnet.to(args.device)
    print("Initialized hypernetwork (decoder) with saved checkpoint:", args.hnet_ckpt_name)

    # freeze the hypernetwork which decodes the conditional embedding that we are optimizing
    for p in hnet.parameters():
        p.requires_grad = False
    # set to eval mode 
    hnet.eval()
    print("Freezed hypernetwork parameters.")
    
    dataset_config = data_configs.separate_embedding_dataset_configs(args)
    dataset = SeparateEmbeddings(dataset_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    print("Loaded dataset for OOD image encoder.")

    optimizer = torch.optim.Adam(embedding.parameters(), lr=args.learning_rate)
    criterion = ClipLoss(args)

    image_embed_dim = args.image_embed_dim
    logit_scale = torch.tensor(math.log(args.logit_scale)).to(args.device)

    bar = tqdm(total=int(args.num_epochs * len(loader)))
    store = {}

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for idx, (image_embeddings, text_embeddings) in enumerate(loader):
            image_embeddings = image_embeddings.to(args.device)
            text_embeddings = text_embeddings.to(args.device)

            optimizer.zero_grad()
            cond_emb = embedding(torch.tensor([0]).to(args.device))
            pred_weight, pred_bias = hnet(cond_emb, image_embed_dim, normalize_output=True, nolookup=True)

            pred_weight = pred_weight.squeeze(0)
            pred_bias = pred_bias.squeeze(0)
            mapped_text_embeddings = text_embeddings @ pred_weight.T + pred_bias

            loss, corrects = criterion.compute_loss_and_accuracy(logit_scale, image_embeddings, mapped_text_embeddings)
            
            correct += corrects
            total += image_embeddings.shape[0]
            accuracy = round(correct/total * 100, 2)

            loss.backward()
            optimizer.step()

            running_loss = round(loss.item(), 2)
            bar.set_description(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss}, Accuracy: {accuracy}%")
            bar.update(1)
        
        store[f"epoch_{epoch+1}"] = {"model": embedding.state_dict(), "loss": running_loss, "accuracy": accuracy}

    store["config"] = vars(args)
    save_path = os.path.join(args.hnet_ckpt_folder, "ood_attempt_1.pt")
    torch.save(store, save_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random-seed", type=int, default=0)
    # hnet settings
    parser.add_argument("--hnet-ckpt-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper")
    parser.add_argument("--hnet-ckpt-epoch", type=int, default=1)
    parser.add_argument("--hnet-ckpt-name", type=str, default="ie_12_mlp_c_32_norm")
    parser.add_argument("--hnet-cond-emb-dim", type=int, default=32)
    parser.add_argument("--hnet-ckpt-num-ie", type=int, default=12)
    parser.add_argument("--largest-image-dim", type=int, default=1536)
    parser.add_argument("--largest-text-dim", type=int, default=768)
    parser.add_argument("--image-embed-dims", type=str, default="384,768,1024")
    parser.add_argument("--hidden-layer-factors", type=str, default="4,16")
    # OOD image encoder settings
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--image-encoder", type=str, default="flexivit_small.300ep_in1k")
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--feature-dataset", type=str, default="cc3m595k")
    # training settings
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--logit-scale", type=float, default=100.0)

    args = parser.parse_args()
    main(args)
