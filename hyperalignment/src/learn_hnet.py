import os
import sys
import math
import wandb
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from contextlib import suppress
from torch.utils.flop_counter import FlopCounterMode
warnings.simplefilter("ignore")

from training.schedulers import *
from models import ConditionalHyperNetwork
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from data.embedding_datasets import MultiMapperEmbeddings
from data import init_encoder_loader, init_indices_loader


@torch.no_grad()
def predict_params_for_saving(model, info):
    model.eval()
    outputs = []

    for v in info.values():
        weights, biases = model(v["embedding"], None, None, image_embed_dim=v["image_embed_dim"], normalize_output=True, nolookup=True, just_params=True)
        outputs.append([weights.squeeze(0), biases.squeeze(0)])

    return outputs


def recursive_average(avg, data, timestep):
    if timestep == 1 and avg is None:
        avg = torch.zeros_like(avg).to(avg.device)

    avg = (timestep - 1) * avg + data.mean(dim=0)
    avg /= timestep
    return avg 


def run(args, input_config):
    torch.manual_seed(args.random_seed)

    # largest mapper shape
    param_shapes = [[args.largest_image_dim, args.largest_text_dim], [args.largest_image_dim]]
    image_embed_dims = [int(x) for x in args.image_embed_dims.split(",")]

    # load in hyper-network
    kwargs = model_configs.hnet_decoder_configs[args.hnet_decoder_type]
    if args.hnet_decoder_type == "chunked_mlp":
        kwargs["chunk_dim"] = args.chunk_dim

    model = ConditionalHyperNetwork(param_shapes, cond_emb_dim=args.hnet_cond_emb_dim, num_cond_embs=args.num_image_encoders, image_embed_dims=image_embed_dims, kwargs=kwargs).to(args.device)
    print("Hyper-network loaded.")

    c = 0
    for p in model.parameters():
        c += p.numel()
    print(c)
    # sys.exit(0)

    # load in dataset and encoder sampler
    config = data_configs.multi_embedding_dataset_configs[args.feature_dataset]
    config["image_encoder_data"] = input_config

    dataset = MultiMapperEmbeddings(config)
    num_batches = math.ceil(len(dataset) / args.batch_size)
    num_encoder_batches = math.ceil(args.num_image_encoders / args.encoder_batch_size)
    
    indices_loader = init_indices_loader(args, dataset)
    encoder_loader = init_encoder_loader(args, dataset)
    print("Dataset loaded.")

    # optimizer + scheduler + scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == "off":
        scheduler = None
    
    elif args.scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, args.num_epochs * num_batches)
    
    elif args.scheduler == "linear":
        scheduler = const_lr_cooldown(optimizer, args.learning_rate, args.warmup_steps, args.num_epochs * num_batches, args.cooldown_steps)

    scaler = torch.cuda.amp.GradScaler() if args.device == "cuda" else None
    autocast = torch.cuda.amp.autocast if args.device == "cuda" else suppress

    ckpt_save_folder = os.path.join(args.checkpoint_folder, args.experiment_type, args.experiment_name, f"seed_{args.random_seed}")
    os.makedirs(ckpt_save_folder, exist_ok=True)

    # logs
    logs = {}
    logit_scale = torch.tensor(np.log(args.logit_scale))
    bar = tqdm(total=args.num_epochs * num_batches)
    encoder_info = {}
    flop_counter = FlopCounterMode(model, display=True, depth=4)

    if args.use_wandb:
        wandb.init(project="hnet-init-scaling", name=args.experiment_name, entity="hyperalignment", config=vars(args))

    # epoch 0 checkpoint
    logs.update({"args": vars(args)})
    dump = {
        "model": model.state_dict(),
        "logs": logs,
    }
    save_path = os.path.join(ckpt_save_folder, f"ckpt_0.pt")
    torch.save(dump, save_path)
    print(f"Checkpoint saved before training as epoch 0.")
    
    # training loop
    with flop_counter: 

        for epoch in range(args.num_epochs):
            total = 0
            correct_store = [0 for _ in range(args.num_image_encoders)]
            running_loss = 0

        # iterate over dataset batches
            for idx in range(num_batches):
                model.train()

                # get the indices of the data samples we want to use
                batch_indices = next(indices_loader)

                # first sample the encoders to use in this step
                encoder_indices, encoder_dims = next(encoder_loader)

                # then get the features made by the sampled encoders
                image_features, text_features = dataset.get_minibatch(batch_indices, encoder_indices, encoder_dims)
                [B, N, D_img] = image_features.shape

                # cast to device
                image_features = image_features.float().to(args.device)
                text_features = text_features.float().to(args.device)

                step = (epoch * num_batches) + idx
                if scheduler is not None:
                    scheduler(step)

                # zero grads
                optimizer.zero_grad()

                # forward pass + loss calculation
                with autocast():
                    if args.cond_type == "indices":
                        loss, corrects = model(cond_id=encoder_indices, image_embed_dim=D_img, normalize_output=args.normalize_output)
                    
                    elif args.cond_type == "features":
                        cond_id = torch.zeros((len(encoder_indices), args.largest_image_dim)).to(args.device)
                        cond_id[:, :args.encoder_dims[0]] = image_features.mean(dim=0)
                        # image_features[:, :, :args.hnet_cond_emb_dim].mean(dim=0)
                        loss, corrects = model(cond_id, image_features, text_features, image_embed_dim=D_img, normalize_output=args.normalize_output, nolookup=True)

                    # mapped_text_features = model.map_features(weights, biases, text_features)
                    # loss, corrects = model.compute_loss(logit_scale, image_features, mapped_text_features, emb_loss=args.emb_loss)
                    assert len(corrects) == N, "Error: number of encoders != number of mappers predicted and evaluated."
                
                # update metric trackers
                running_loss = loss.item()
                total += B
                for j, c in zip(encoder_indices, corrects):
                    correct_store[j] += c
                accuracies = [round(x/total * 100, 2) for x in correct_store]
                accuracies = [str(item)+"*" if jdx in encoder_indices else str(item) for (jdx, item) in enumerate(accuracies)]

                # backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()

                    if args.clip_grad_norm != -1:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                
                else:
                    loss.backward()
                    optimizer.step()
                
                # add to logs
                logs[f"epoch_{epoch+1}"] = {"loss": running_loss}
                for k in range(len(accuracies)):
                    logs[f"epoch_{epoch+1}"][f"acc_{k}"] = accuracies[k]

                # optional wandb logging
                if args.use_wandb:
                    wandb.log(logs[f"epoch_{epoch+1}"], step=step)
            
                # print the result of the current epoch
                bar.update(1)
                bar.set_description(f"Epoch: {epoch+1}, Step: {(epoch * num_batches) + idx+1}")
                bar.set_postfix(logs[f"epoch_{epoch+1}"])

                # update info about embeddings and H-Net
                model.eval()

                for ii, index in enumerate(encoder_indices):
                    if args.cond_type == "indices":
                        encoder_info[index] = {"index": index, "image_embed_dim": D_img, "embedding": model.lookup_embedding_table(index)}

                    elif args.cond_type == "features":
                        encoder_info[index] = {"index": index, "image_embed_dim": D_img, "embedding": cond_id[ii, :].view(1, args.hnet_cond_emb_dim)}
                
        # make sure we have saved info correctly
        assert len(encoder_info.keys()) == args.num_image_encoders, "Something went wrong during storing info for H-Net."
        
        # make sure that we save
        if (epoch+1) in [1, 2, 5, 10, 20, 40, 100, 200] and args.saving:
            model.eval()
            logs.update({"args": args.__dict__})
            dump = {
                "model": model.state_dict(),
                "logs": logs,
                "mapper_params": predict_params_for_saving(model, encoder_info),
                "info": encoder_info
            }
            save_path = os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}.pt")
            torch.save(dump, save_path)
            print(f"Checkpoint saved at epoch {epoch+1}.")
        
    print("All done.")


def main(args):
    full_configs = model_configs.ID_multi_mapper_configs
    num_encoders_ablation = [9, 12, 15, 18, 21, 24, 27, 30]
    assert args.num_image_encoders in num_encoders_ablation, "Incompatible number selected for ablation!"

    num_encoders = args.num_image_encoders
    q = num_encoders // 3
    config = {k:full_configs[k][:q] for k in full_configs.keys()}
    args.encoder_batch_size = q

    print(f"Started run with num_image_encoders: {num_encoders}")
    run(args, input_config=config)
    print(f"Finished run with num_image_encoders: {num_encoders}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # global args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
    parser.add_argument("--experiment-type", type=str, default="multi_mapper")
    parser.add_argument("--experiment-name", type=str, default="ie_12_mlp_c_32_norm_ft_ep_10_lr1e-2")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    parser.add_argument("--cond-type", type=str, default="features", choices=["indices", "features"])
    # model args
    parser.add_argument("--feature-dataset", type=str, default="cc3m595k")
    parser.add_argument("--largest-image-dim", type=int, default=1024)
    parser.add_argument("--largest-text-dim", type=int, default=768)
    parser.add_argument("--image-embed-dims", type=str, default="384,768,1024")
    parser.add_argument("--hnet-cond-emb-dim", type=int, default=32)
    parser.add_argument("--hnet-decoder-type", type=str, default="mlp")
    parser.add_argument("--num-image-encoders", type=int, default=12)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--normalize-output", type=bool, default=True)
    parser.add_argument("--rescale-factor", type=float, default=0.0)
    parser.add_argument("--emb-loss", type=bool, default=False)
    parser.add_argument("--chunk-dim", type=int, default=256)
    # training args
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="off")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--cooldown-steps", type=int, default=500)
    parser.add_argument("--flop-counter", type=str, default="custom")
    parser.add_argument("--clip-grad-norm", type=float, default=-1)
    parser.add_argument("--saving", type=bool, default=True)
    parser.add_argument("--scaling-ablation", type=bool, default=False)

    args = parser.parse_args()

    print("Experiment name:", args.experiment_name)
    print("------------------------------------------------------------")
    print("Num epochs:", args.num_epochs)
    print("Scheduler:", args.scheduler)
    print("Cond emb dim:", args.hnet_cond_emb_dim)
    print("Checkpoints will be saved:", args.saving)
    print("Weights are normalized when predicted:", args.normalize_output)

    main(args)
