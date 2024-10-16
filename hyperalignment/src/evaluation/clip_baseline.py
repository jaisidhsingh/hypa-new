import sys
import clip
import torch
from torch.utils.data import DataLoader

from src.data import ImageCaptionDataset
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.evaluation.evaluate import coco_captions_eval
from src.models import *

# from hypnettorch.hnets import HMLP
# from hypnettorch.mnets import MLP

device = "cuda"
# clip_model, preprocess = clip.load("ViT-B/16", device="cuda")

# cfg = data_configs.image_caption_dataset_configs["mscoco_val"]
# cfg["feature_dataset"] = "mscoco_val"
# cfg["transform"] = preprocess

# dataset = ImageCaptionDataset(cfg)
# loader = DataLoader(dataset, batch_size=1024, pin_memory=True, collate_fn=dataset.collate_fn)

# logs = coco_captions_eval(clip_model, loader, progress_bar=True, using_clip=True)
# print(logs)

# print("CLIP -->", logs["accuracy"])
# print("----------------------------")
# print(" ")
# del clip_model

"""
for seed in range(5):
    torch.manual_seed(seed)
    for epoch in [1, 5, 10, 20]:
        # joint first
        params = torch.load(f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/id_vitr/joint/hmlp-1_id_vitr_ls-100.0.cc3m595k_epochs-20/seed_{seed}/ckpt_{epoch}.pt")["mapper_params"]
        weights = params[0]
        #print(weights[1][:5])

        model = CustomVLM("vit_base_patch16_224", "sentence-t5-base")
        model.mapper = MlpMapper(768, [], 768).to(device)
        model.mapper.layers[0].weight.data = weights[0]
        model.mapper.layers[0].bias.data = weights[1]

        cfg = data_configs.image_caption_dataset_configs["mscoco_val"]
        cfg["feature_dataset"] = "mscoco_val"
        cfg["transform"] = model.image_encoder.transform
        dataset = ImageCaptionDataset(cfg)
        loader = DataLoader(dataset, batch_size=1024, pin_memory=True, collate_fn=dataset.collate_fn)

        joint_logs = coco_captions_eval(model, loader, progress_bar=True, using_clip=False)
        print("Joint - seed", seed, "epoch", epoch, "vit_base_patch16_224 gives -->", joint_logs["accuracy"])
"""


# main_net = MLP(n_in=768, hidden_layers=[], n_out=768, no_weights=True)
# hnet = HMLP(
# 		main_net.param_shapes, uncond_in_size=0,
# 		cond_in_size=768, layers=[],
# 		num_cond_embs=6
# ).to(device)
# ckpt = torch.load("/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/old_id/joint/hnet_id_vitr_ls-100.cc3m300k_epochs-20/ckpt_1.pt")["model"]
# hnet.load_state_dict(ckpt)
# weights = hnet(cond_id=0)

i = int(sys.argv[1])

saved_params = torch.load("/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/id_vitr/id_vitr/joint/hmlp-1_cond_emb-768.cc3m595k_epochs-1/seed_0/ckpt_1.pt")["mapper_params"]
weights = saved_params[i]

# then separate

configs = model_configs.ID_experiment_configs["id_vitr"]
image_encoder = configs["image_encoders"][i]

model = CustomVLM(image_encoder, "sentence-t5-base")
model.mapper = MlpMapper(768, [], 768).to(device)

model.mapper.layers[0].weight.data = weights[0]
model.mapper.layers[0].bias.data = weights[1]

cfg = data_configs.image_caption_dataset_configs["mscoco_val"]
cfg["feature_dataset"] = "mscoco_val"
cfg["transform"] = model.image_encoder.transform

dataset = ImageCaptionDataset(cfg)
loader = DataLoader(dataset, batch_size=1024, pin_memory=True, collate_fn=dataset.collate_fn, shuffle=False)


if sys.argv[2] == "sep":
    model.mapper.load_state_dict(
        torch.load(
            f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/id_vitr/separate/id_vitr_ie-0_te-0.cc3m595k_id_vitr_raw_epochs-40/seed_0/ckpt_1.pt"
        )["model"]
    )


if sys.argv[2] == "jft":
    model.mapper.load_state_dict(
        torch.load(
            f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/id_vitr/id_vitr/joint/hmlp-1_cond_emb-768.cc3m595k_epochs-1/seed_0/ie-0_te-0/ft_ckpt_1.pt"
        )["model"]
    )

# print(model.mapper.layers[0].bias.data[:5])

# (image, caption) = dataset[0]
# imf = model.encode_image(image.unsqueeze(0).to(device))
# print(imf.flatten()[:5])

separate_logs = coco_captions_eval(model, loader, progress_bar=True, using_clip=False)
print(separate_logs)
# print("Separate - seed", seed, "epoch", epoch, "vit_base_patch16_224 gives -->", separate_logs["accuracy"])
