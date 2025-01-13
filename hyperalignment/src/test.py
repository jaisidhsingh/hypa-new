import os
import sys
import math
import argparse
import warnings
from tqdm import tqdm
from contextlib import suppress
warnings.simplefilter("ignore")
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from models import *
from training.loss_functions import ClipLoss
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from utils.backward_flops import FlopCounterMode
from utils.check_cka import CKA
from utils.perm_tests import align_features, compute_cost_matrix


def sanity_check(args):
    print(args.image_encoder)

    prefix = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results"

    ood_fts_path = f"{prefix}/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_{args.image_embed_dim}/{args.image_encoder}/memmap.npy"
    ood_fts = np.array(np.memmap(ood_fts_path, dtype="float32", mode="r", shape=(595375,384)))

    id_fts_paths = [f"{prefix}/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_{args.image_embed_dim}/{ie}/memmap.npy" for ie in model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"][:4]]
    id_fts_list = [np.array(np.memmap(path, dtype="float32", mode="r", shape=(595375, 384))) for path in id_fts_paths]

    P = align_features(ood_fts, id_fts_list, return_perm=True)
    return torch.from_numpy(P)

    # # Feature-wise L2 distances
    # for item in id_fts_list:
    #     old = compute_cost_matrix(ood_fts, item)
    #     new = compute_cost_matrix(aligned_ood_fts, item)

    #     print(np.diag(old).mean())
    #     print(np.diag(new).mean()) 



def cka_based_sim_weighting(args):
    print(args.image_encoder)

    prefix = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results"

    ood_fts_path = f"{prefix}/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_{args.image_embed_dim}/{args.image_encoder}/memmap.npy"
    ood_fts = np.array(np.memmap(ood_fts_path, dtype="float32", mode="r", shape=(595375,384)))[:10000, :]

    id_fts_paths = [f"{prefix}/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_{args.image_embed_dim}/{ie}/memmap.npy" for ie in model_configs.ID_experiment_configs["multi_mapper"][args.image_embed_dim]["image_encoders"][:4]]
    id_fts_list = [np.array(np.memmap(path, dtype="float32", mode="r", shape=(595375, 384)))[:10000, :] for path in id_fts_paths]

    cka = CKA("cuda")

    aligned_fts = align_features(ood_fts, id_fts_list)
    aligned_fts = torch.from_numpy(aligned_fts).cuda()
    ood_fts = torch.from_numpy(ood_fts).cuda()

    for item in id_fts_list:
        old_cka = cka.linear_CKA(ood_fts, torch.from_numpy(item).cuda())
        print(old_cka)


def main(args):
    torch.manual_seed(args.random_seed)

    # P = sanity_check(args)

    param_shapes = [[args.largest_image_dim, args.largest_text_dim], [args.largest_image_dim]]
    image_embed_dims = [int(x) for x in args.image_embed_dims.split(",")]

    # embedding = nn.Embedding(1, args.hnet_cond_emb_dim)
    
    # for param in embedding.parameters():
    #     param.requires_grad = True
    # embedding = embedding.to(args.device)
    # print("Initialized embedding to auto-decode.")

    decoder_type = args.hnet_ckpt_name.split("_")[2]
    kwargs = model_configs.hnet_decoder_configs[decoder_type]

    hnet = ConditionalHyperNetwork(
        param_shapes, cond_emb_dim=args.hnet_cond_emb_dim,
        num_cond_embs=args.hnet_ckpt_num_ie, image_embed_dims=image_embed_dims, kwargs=kwargs 
    )

    ckpt_path = os.path.join(args.hnet_ckpt_folder, args.hnet_ckpt_name, f"seed_{args.random_seed}", f"ckpt_{args.hnet_ckpt_epoch}.pt")
    hnet.load_state_dict(torch.load(ckpt_path)["model"])
    hnet = hnet.to(args.device)
    print("Initialized hypernetwork (decoder) with saved checkpoint:", args.hnet_ckpt_name)

    # freeze the hypernetwork which decodes the conditional embedding that we are optimizing
    # for p in hnet.parameters():
    #     p.requires_grad = False #True
    # # set to eval mode 
    hnet.train()
    # print("Freezed hypernetwork parameters.")

    # Initialise the embedding to be learnt as the avg of the hnet's embeddings
    # embedding.weight.data = hnet.cond_embs.weight.data[:4, :].mean(dim=0).unsqueeze(0)
    
    dataset_config = data_configs.separate_embedding_dataset_configs(args)
    dataset = SeparateEmbeddings(dataset_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    print("Loaded dataset for OOD image encoder.")


    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.learning_rate)
    criterion = ClipLoss(args)

    image_embed_dim = args.image_embed_dim
    logit_scale = torch.tensor(math.log(args.logit_scale)).to(args.device)

    store = {}
    hnet_cond_emb_dim = int(args.hnet_ckpt_name.split("_")[4])
    flop_counter = FlopCounterMode(hnet)
    bar = tqdm(total=int(len(loader)))
    train = True

    def set_model(model, train):
        if train:
            model.train()
        else:
            model.eval()

    set_model(hnet, train)
    with flop_counter:

        running_loss = 0.0
        correct, total = 0, 0
        
        for idx, (image_embeddings, text_embeddings) in enumerate(loader):
            if idx >= args.train_steps:
                train = False
                set_model(hnet, train)

            image_embeddings = image_embeddings.to(args.device)
            text_embeddings = text_embeddings.to(args.device)

            if train:
                optimizer.zero_grad()

            cond_emb = image_embeddings[:, :hnet_cond_emb_dim].mean(dim=0).view(1, hnet_cond_emb_dim)
            pred_weight, pred_bias = hnet(cond_emb, image_embed_dim, normalize_output=True, nolookup=True)
            
            pred_weight = pred_weight.squeeze(0)
            pred_bias = pred_bias.squeeze(0)

            if idx == 0:
                store["step_0"] = {"weight": pred_weight, "bias": pred_bias}

            mapped_text_embeddings = text_embeddings @ pred_weight.T + pred_bias

            loss, corrects = criterion.compute_loss_and_accuracy(logit_scale, image_embeddings, mapped_text_embeddings)
            
            correct += corrects
            total += image_embeddings.shape[0]
            accuracy = round(correct/total * 100, 2)

            if train:
                loss.backward()
                optimizer.step()

            running_loss = round(loss.item(), 2)
            bar.set_description(f"Step {idx+1}/{args.total_steps}, Loss: {running_loss}, Accuracy: {accuracy}%")
            bar.update(1)

            if idx+1 % 500 == 0:
                store[f"step_{idx+1}"] = {"weight": pred_weight, "bias": pred_bias}

            if idx == args.total_steps-1:
                break
            
    hnet.eval() 
    pred_weight, pred_bias = hnet(cond_emb, image_embed_dim, normalize_output=True, nolookup=True)
    store[f"epoch_{0+1}"] = {"mapper_params": [pred_weight.squeeze(0), pred_bias.squeeze(0)], "loss": 0, "accuracy": 0}

    store["config"] = vars(args)
    args.save_path = args.image_encoder + "_ood_ft.pt"

    save_folder = os.path.join(args.hnet_ckpt_folder, "ood_attempts")
    os.makedirs(save_folder, exist_ok=True)

    torch.save(store, os.path.join(save_folder, args.save_path))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="x")
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
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logit-scale", type=float, default=100.0)

    args = parser.parse_args()
    main(args)
