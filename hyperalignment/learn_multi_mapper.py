import os
import math
import wandb
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from contextlib import suppress
warnings.simplefilter("ignore")

from src.data.embedding_datasets import MultiMapperEmbeddings
from src.data import init_encoder_loader, init_indices_loader
from src.models import MultiMapperHypernet
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.training.schedulers import *
from src.utils import get_hypnet_flops
from src.utils.backward_flops import FlopCounterMode


@torch.no_grad()
def predict_params_for_saving(model, info):
    model.eval()
    outputs = []
    for k, v in info.items():
        weights, biases = model(cond_id=v, image_embed_dim=k)
        for (w, b) in zip(weights.cpu(), biases.cpu()):
            outputs.append([w, b])

    return outputs


def run(args, input_config):
    torch.manual_seed(args.random_seed)

    # largest mapper shape
    param_shapes = [[args.largest_image_dim, args.largest_text_dim], [args.largest_image_dim]]
    image_embed_dims = [int(x) for x in args.image_embed_dims.split(",")]
    hidden_layer_factors = [int(x) for x in args.hidden_layer_factors.split(",")]
    
    # make an easy switch to linear
    if hidden_layer_factors == [0]:
        hidden_layer_factors = []

    # load in hyper-network
    model = MultiMapperHypernet(
        param_shapes, cond_emb_dim=args.hnet_cond_emb_dim,
        num_cond_embs=args.num_image_encoders, image_embed_dims=image_embed_dims,
        hidden_layer_factors=hidden_layer_factors, rescale_factor=args.rescale_factor
    ).to(args.device)
    print("Hyper-network loaded.")

    if args.flop_counter == "calflops":
        hnet_flops = get_hypnet_flops(model, kwargs={"cond_id": [0], "image_embed_dim": 768})
        print(f"FLOPs for hyper-network (one encoder only): {round(hnet_flops[0], 2)} x 10^{math.log10(hnet_flops[1])}")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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

    flop_counter = FlopCounterMode(model) if args.flop_counter == "custom" else suppress

    if args.use_wandb:
        wandb.init(project="hnet-init-scaling", name=args.experiment_name, entity="hyperalignment", config=vars(args))

    # training loop
    for epoch in range(args.num_epochs):
        total = 0
        correct_store = [0 for _ in range(args.num_image_encoders)]
        running_loss = 0

        # iterate over dataset batches
        with flop_counter: 
            for idx in range(num_batches):

                # get the indices of the data samples we want to use
                batch_indices = next(indices_loader)

                # first sample the encoders to use in this step
                encoder_indices, encoder_dims = next(encoder_loader)

                # collect the indices and dims as info to be used later
                if len(encoder_info.keys()) < num_encoder_batches:
                    encoder_info[encoder_dims[0]] = encoder_indices

                # then get the features made by the sampled encoders
                image_features, text_features = dataset.get_minibatch(batch_indices, encoder_indices, encoder_dims)
                [B, N, D_img] = image_features.shape

                # cast to device
                image_features = image_features.float().to(args.device)
                text_features = text_features.float().to(args.device)

                if scheduler is not None:
                    step = (epoch * num_batches) + idx
                    scheduler(step)

                # zero grads
                optimizer.zero_grad()

                # forward pass + loss calculation
                with autocast():
                    weights, biases = model(cond_id=encoder_indices, image_embed_dim=D_img, normalize_output=args.normalize_output)
                    mapped_text_features = model.map_features(weights, biases, text_features)
                    loss, corrects = model.compute_loss(logit_scale, image_features, mapped_text_features)
                    assert len(corrects) == N, "Error: number of encoders != number of mappers predicted and evaluated."
                
                # update metric trackers
                running_loss += loss.item()
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
                logs[f"epoch_{epoch+1}"] = {"avg_loss": round(running_loss / (idx+1), 2), "accuracies": accuracies}

                # optional wandb logging
                if args.use_wandb:
                    wandb.log(logs[f"epoch_{epoch+1}"], step=step)
            
                # print the result of the current epoch
                bar.update(1)
                bar.set_description(f"Epoch: {epoch+1}, Step: {(epoch * num_batches) + idx+1}")
                bar.set_postfix(logs[f"epoch_{epoch+1}"])
            
            if epoch == 0:
                saved_flop_counter = deepcopy(flop_counter)
                print(f"FLOPs for one training epoch: {saved_flop_counter.results}")
                flop_counter = suppress

        # make sure that we save
        if (epoch+1) in [1, 2, 5, 10, 20, 40, 100, 200] and args.saving:
            logs.update({"args": args.__dict__})
            dump = {
                "optimizer": optimizer.state_dict(),
                "logs": logs,
                "mapper_params": predict_params_for_saving(model, encoder_info),
                "one_epoch_flop_count": saved_flop_counter.results
            }
            save_path = os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}.pt")
            torch.save(dump, save_path)
            print(f"Checkpoint saved at epoch {epoch+1}.")
        
    print("All done.")


def main(args):
    full_configs = model_configs.ID_multi_mapper_configs
    num_encoders_ablation = [12, 15, 18, 24, 30]
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
    parser.add_argument("--experiment-name", type=str, default="multi_mapper_test_0_fixed")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    # model args
    parser.add_argument("--feature-dataset", type=str, default="cc3m595k")
    parser.add_argument("--largest-image-dim", type=int, default=1536)
    parser.add_argument("--largest-text-dim", type=int, default=768)
    parser.add_argument("--image-embed-dims", type=str, default="384,768,1024")
    parser.add_argument("--hidden-layer-factors", type=str, default="4,16")
    parser.add_argument("--hnet-cond-emb-dim", type=int, default=8)
    parser.add_argument("--num-image-encoders", type=int, default=30)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--normalize-output", type=bool, default=False)
    parser.add_argument("--rescale-factor", type=float, default=10.0)
    # training args
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=10)
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
    print("Hyper-net decoder hidden dims:", [int(x)*args.hnet_cond_emb_dim for x in args.hidden_layer_factors.split(",")])
    print("Cond emb dim:", args.hnet_cond_emb_dim)
    print("Scaled initiation is on:", args.rescale_factor != 0.0)
    print("Weights are normalized when predicted:", args.normalize_output)
    print("Scheduler:", args.scheduler)

    main(args)
