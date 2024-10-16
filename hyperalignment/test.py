import torch
from torch.utils.data import DataLoader
import sys
from src.models import MultiMapperHypernet, CustomVLM, MlpMapper
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.data.image_caption_datasets import ImageCaptionDataset
from evaluate_mscoco import coco_captions_eval, coco_captions_eval2


device = "cuda"
# param_shapes = [[1024, 768], [1024]]
# model = MultiMapperHypernet(
#     param_shapes=param_shapes, cond_emb_dim=8,
#     num_cond_embs=30, image_embed_dims=[384, 768, 1024],
#     hidden_layer_factors=[]
# ).to(device)

ep = int(sys.argv[2])
num_ie = int(sys.argv[3])
exp_name = sys.argv[1]
path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/mlp-4-16_multi-mapper_num_ie-{num_ie}.cc3m595k_epochs-20/seed_0/ckpt_{ep}.pt"
path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/{exp_name}/seed_0/ckpt_{ep}.pt"
# path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/mlp-4-16_multi_mapper_num_ie-30.cc3m595k_epochs-20/seed_0/ckpt_{ep}.pt"
ckpt = torch.load(path)
params = ckpt["mapper_params"]
logs = ckpt["logs"]

# model.load_state_dict(ckpt)
# weights, biases = model(cond_id=[0], image_embed_dim=384)

q = num_ie // 3
print("Previous saved param shape", params[q-1][0].shape) # 384

[weights, biases] = params[q] # vit_base_patch16_224

print(weights.shape, biases.shape)

image_encoder = "vit_base_patch16_224"
text_encoder = "sentence-t5-base"
model = CustomVLM(image_encoder, text_encoder)
model.mapper = MlpMapper(768, [], 768).to(device)
model.mapper.layers[0].weight.data = weights.to(device)
model.mapper.layers[0].bias.data = biases.to(device)

model.mapper = model.mapper.to(device)

config = data_configs.image_caption_dataset_configs["mscoco_val"]
config.update({"transform": model.image_encoder.transform, "feature_dataset": "mscoco_val"})
dataset = ImageCaptionDataset(config)
loader = DataLoader(dataset, batch_size=1024, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False)

# accuracy = coco_captions_eval(model, loader)
accuracy = coco_captions_eval(model, loader)
print(accuracy)
