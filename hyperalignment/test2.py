import torch
from torch.utils.data import DataLoader
import os
import sys
from evaluate_mscoco import coco_captions_eval
import clip
from src.models import CustomVLM, MlpMapper
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.data.image_caption_datasets import ImageCaptionDataset
from evaluate_mscoco import coco_captions_eval, coco_captions_eval2


device = "cuda"

dim = int(sys.argv[1])
ep = int(sys.argv[2])

models = model_configs.ID_multi_mapper_configs[dim][0]
models = [models]

for model_name in models:
    path = os.path.join(
        "/home/mila/s/sparsha.mishra/scratch/hypa/checkpoints/multi_mapper/separate",
        model_name,
        "seed_0",
        f"ckpt_{ep}.pt"
    )
    ckpt = torch.load(path)["model"]

    model = CustomVLM(image_encoder_name=model_name, text_encoder_name="sentence-t5-base")
    model.mapper = MlpMapper(768, [], dim).to(device)
    # sys.exit(0)
    model.mapper.load_state_dict(ckpt)
    model.mapper.eval()
    # model.mapper = model.mapper.to(device)
    
    # model, preprocess = clip.load("ViT-B/16", device="cuda")

    config = data_configs.image_caption_dataset_configs["mscoco_val"]
    config.update({"transform": model.image_encoder.transform, "feature_dataset": "mscoco_val"})
    # config.update({"transform": preprocess, "feature_dataset": "mscoco_val"})
    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False)

    accuracy = coco_captions_eval(model, loader)
    # accuracy = coco_captions_eval(model, loader, using_clip=True)
    print(model_name)
    # print("CLIP vit-b/16")
    print(accuracy)
    print("-------")
    print("  ")

# path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/mlp-4-16_multi-mapper_num_ie-{num_ie}.cc3m595k_epochs-20/seed_0/ckpt_{ep}.pt"
# path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/{exp_name}/seed_0/ckpt_{ep}.pt"
# # path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/mlp-4-16_multi_mapper_num_ie-30.cc3m595k_epochs-20/seed_0/ckpt_{ep}.pt"
# ckpt = torch.load(path)
# params = ckpt["mapper_params"]
# logs = ckpt["logs"]

# # model.load_state_dict(ckpt)
# # weights, biases = model(cond_id=[0], image_embed_dim=384)

# q = num_ie // 3
# print("Previous saved param shape", params[q-1][0].shape) # 384

# [weights, biases] = params[q] # vit_base_patch16_224

# print(weights.shape, biases.shape)

# image_encoder = "vit_base_patch16_224"
# text_encoder = "sentence-t5-base"
# model = CustomVLM(image_encoder, text_encoder)
# model.mapper = MlpMapper(768, [], 768).to(device)
# model.mapper.layers[0].weight.data = weights.to(device)
# model.mapper.layers[0].bias.data = biases.to(device)

# model.mapper = model.mapper.to(device)

# config = data_configs.image_caption_dataset_configs["mscoco_val"]
# config.update({"transform": model.image_encoder.transform, "feature_dataset": "mscoco_val"})
# dataset = ImageCaptionDataset(config)
# loader = DataLoader(dataset, batch_size=1024, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False)

# accuracy = coco_captions_eval(model, loader)
# print(accuracy)
