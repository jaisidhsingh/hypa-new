import os
import sys
import math
import argparse
import warnings
from tqdm import tqdm
from contextlib import suppress
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from models import *
from training.loss_functions import ClipLoss
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from torch.utils.flop_counter import FlopCounterMode
from train_ape import evaluate_mapper


def adapt(args):
    torch.manual_seed(args.random_seed)

    param_shapes = [[args.largest_image_dim, args.largest_text_dim], [args.largest_image_dim]]
    image_embed_dims = [int(x) for x in args.image_embed_dims.split(",")]

    decoder_type = "mlp"
    kwargs = model_configs.hnet_decoder_configs[decoder_type]

    hnet = ConditionalHyperNetwork(
        param_shapes, cond_emb_dim=args.hnet_cond_emb_dim,
        num_cond_embs=args.hnet_ckpt_num_ie, image_embed_dims=image_embed_dims, kwargs=kwargs 
    )

    ckpt_path = os.path.join(args.hnet_ckpt_folder, args.hnet_ckpt_name, f"seed_{args.random_seed}", f"ckpt_{args.hnet_ckpt_epoch}.pt")
    hnet.load_state_dict(torch.load(ckpt_path)["model"])
    hnet = hnet.to(args.device)
    print("Initialized hypernetwork (decoder) with saved checkpoint:", args.hnet_ckpt_name)

    if args.mode == "proj":
        for n, p in hnet.named_parameters():
            if "in_proj" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    # set to eval mode 
    hnet.eval()
    print("Froze hypernetwork parameters.")
    
    dataset_config = data_configs.separate_embedding_dataset_configs(args)
    dataset = SeparateEmbeddings(dataset_config, split="train", args=args)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    print("Loaded dataset for OOD image encoder.")

    optimizer = torch.optim.AdamW(hnet.parameters(), lr=args.learning_rate)
    image_embed_dim = args.image_embed_dim

    bar = tqdm(total=int(args.num_epochs * len(loader)))
    store = {}

    flop_counter = FlopCounterMode(hnet, display=True, depth=2)

    with flop_counter:
        for epoch in range(args.num_epochs):
            running_loss = 0.0
            correct, total = 0, 0

            for idx, (image_embeddings, text_embeddings) in enumerate(loader):
                image_embeddings = image_embeddings.to(args.device)
                D_img = image_embeddings.shape[-1]
                text_embeddings = text_embeddings.to(args.device)

                optimizer.zero_grad()

                cond_id = torch.zeros((1, args.largest_image_dim)).to(args.device)
                cond_id[:, :D_img] = image_embeddings.mean(dim=0).unsqueeze(0)
                loss, corrects = hnet(cond_id, image_embeddings.unsqueeze(1), text_embeddings, image_embed_dim, normalize_output=True, nolookup=True)

                correct += corrects[0]
                total += image_embeddings.shape[0]
                accuracy = round(correct/total * 100, 2)

                loss.backward()
                optimizer.step()

                running_loss = round(loss.item(), 2)
                bar.set_description(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss}, Accuracy: {accuracy}%")
                bar.update(1)

                if idx == args.break_point:
                    break
        
    pred_weight, pred_bias = hnet(cond_id, image_embeddings.unsqueeze(1), text_embeddings, image_embed_dim, normalize_output=True, nolookup=True, just_params=True)
    store[f"epoch_{epoch+1}"] = {"mapper_params": [pred_weight.squeeze(0), pred_bias.squeeze(0)], "loss": running_loss, "accuracy": accuracy}

    store["config"] = vars(args)
    args.save_path = args.image_encoder + "_full_ood.pt"

    save_folder = os.path.join(args.hnet_ckpt_folder, "icml_ood")
    os.makedirs(save_folder, exist_ok=True)

    torch.save(store, os.path.join(save_folder, args.save_path))

    print("Done!")
    return pred_weight, pred_weight, dataset


def ft(args, w, b, dataset):
    loader = DataLoader(dataset, batch_size=args.ft_batch_size, num_workers=args.num_workers, pin_memory=True)
    model = MLP(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
    model.layers[0].weight.data = w
    model.layers[0].bias.data = b
    model.train()

    criterion = ClipLoss(args)
    optimizer = torch.optim.AdamW(model.parameters, lr=args.ft_lr)

    flop_counter = FlopCounterMode(model, display=True, depth=2)
    bar = tqdm(total=len(loader))
    logit_scale = torch.tensor(np.log(100.0)).to(args.device)

    running_loss = 0
    total = 0
    for idx, (image_features, text_features) in enumerate(loader):
        bs = image_features.shape[0]
        image_features = image_features.float().to(args.device).view(bs, args.image_embed_dim)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = text_features.float().to(args.device).view(bs, args.text_embed_dim)
        
        optimizer.zero_grad()
        mapped_features = model(text_features)
        loss, corrects = criterion.compute_loss_and_accuracy(logit_scale, image_features, mapped_features)

        accuracy = round(corrects / total * 100, 2)
        running_loss = loss.item()

        loss.backward()
        optimizer.step()

        bar.set_postfix({"Acc": accuracy, "loss": running_loss})
        bar.update(1)
    
    model.eval()
    return model


def main(args):
    w, b, dataset = adapt(args)
    model = ft(args, w, b, dataset)
    acc, loss = evaluate_mapper(args, model)
    print(acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="x")
    # hnet settings
    parser.add_argument("--hnet-ckpt-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper")
    parser.add_argument("--hnet-ckpt-epoch", type=int, default=10)
    parser.add_argument("--hnet-ckpt-name", type=str, default="hnet_30-10_fmlp_c-32_bs-512_lr-1e-2")
    parser.add_argument("--hnet-cond-emb-dim", type=int, default=32)
    parser.add_argument("--hnet-ckpt-num-ie", type=int, default=30)
    parser.add_argument("--largest-image-dim", type=int, default=1024)
    parser.add_argument("--largest-text-dim", type=int, default=768)
    parser.add_argument("--image-embed-dims", type=str, default="384,768,1024")
    # OOD image encoder settings
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--image-encoder", type=str, default="eva02_small_patch14_224.mim_in22k")
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--feature-dataset", type=str, default="cc3m558k")
    # training settings
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--break-point", type=float, default=100)
    parser.add_argument("--data-scaling", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--ft-batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--ft-lr", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)
