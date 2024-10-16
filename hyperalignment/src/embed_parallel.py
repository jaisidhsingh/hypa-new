import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from PIL import Image
from torch import Tensor
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.simplefilter("ignore")

from configs.model_configs import model_configs
from configs.data_configs import data_configs
from models import TextEncoder, ImageEncoder
from data import ImageCaptionDataset


@torch.no_grad()
def embed_images(args, model_names):
    save_folder_name = f"{args.feature_dataset}_{args.experiment_type}" if args.path_extension == "" else f"{args.feature_dataset}_{args.experiment_type}_{args.path_extension}"
    save_folder = os.path.join(args.results_folder, "image_embeddings", args.experiment_type, save_folder_name, f"dim_{args.image_embed_dim}")
    os.makedirs(save_folder, exist_ok=True)

    # save_path = os.path.join(save_folder, f"dim_{args.image_embed_dim}.pt")

    autocast = torch.cuda.amp.autocast
    store = {}
    store[args.image_embed_dim] = {}

    N = len(model_names)

    for encoder_name in model_names:
        print(encoder_name)
        folder_for_encoder = os.path.join(save_folder, encoder_name)
        os.makedirs(folder_for_encoder, exist_ok=True)

        image_encoder = ImageEncoder(model_name=encoder_name, device=args.device)
        torch.compile(image_encoder)

        kwargs = {"feature_dataset": args.feature_dataset, "transform": image_encoder.transform}
        dataset_config = data_configs.image_caption_dataset_configs[kwargs["feature_dataset"]]
        kwargs.update(dataset_config)

        if args.feature_dataset == "cc3m300k":
            kwargs.update({"path_appendage": f"{data_configs.STORE}/cc3m300k/images"})

        elif args.feature_dataset == "cc3m595k":
            kwargs.update({"path_appendage": f"{data_configs.STORE}/LLaVA-CC3M-Pretrain-595K/llava_cc3m595k_images"})
            kwargs["caption_type"] = args.extension

        dataset = ImageCaptionDataset(kwargs)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)

        features = torch.zeros(len(dataset), args.image_embed_dim)
        bar = tqdm(total=len(loader))

        for i, (images, captions) in enumerate(loader):
            batch_size = images.shape[0]
            # removing captions from RAM as they are not used here
            del captions

            images = images.float().to(args.device)
            with autocast():
                image_features = image_encoder.encode_image(images)
            features[i * batch_size: (i+1) * batch_size] = image_features.cpu()

            # np.save(os.path.join(folder_for_encoder, f"emb_batch_{i}.npy"), image_features.cpu().numpy())

            bar.update(1)
            bar.set_postfix({"encoder": encoder_name, "dim": args.image_embed_dim})

        bar.close()
        np.save(os.path.join(folder_for_encoder, "embeddings.npy"), features.cpu().numpy())
        # store[args.image_embed_dim][encoder_name] = features

    # torch.save(store, save_path)
    print(f"{features.shape[0]} images encoded across {N} image encoders")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-type", type=str, default="multi_mapper")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-embed-dim", type=int, default=1024)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--results-folder", type=str, default=data_configs.embedding_store_root)
    parser.add_argument("--feature-dataset", type=str, default="cc3m595k")
    parser.add_argument("--mode", type=str, default="images")
    parser.add_argument("--extension", type=str, default="raw")
    parser.add_argument("--path-extension", type=str, default="30_ie")
    parser.add_argument("--encoder-indices", type=str, default="0,1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--parallel", type=bool, default=False)
    args = parser.parse_args()

    # easy access for models of experiments
    models_to_exp_map = model_configs.ID_experiment_configs
    # sanity check for our results folder
    os.makedirs(args.results_folder, exist_ok=True)

    encoders = [models_to_exp_map[args.experiment_type][args.image_embed_dim]["image_encoders"][int(i)] for i in args.encoder_indices.split(",")]

    # if not args.parallel:
    embed_images(args, encoders)
    # else:
    #     out = Parallel(n_jobs=-1, prefer="threads")(
    #         delayed(embed_images)(args, [e]) for e in encoders
    #     )
