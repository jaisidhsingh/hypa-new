import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hyperalignment.src.evaluation.evaluate import coco_captions_eval
from src.models import MlpMapper, CustomVLM
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.data.image_caption_datasets import ImageCaptionDataset


class MultiMapperHypernet(nn.Module):
    def __init__(self, param_shapes, cond_emb_dim, num_cond_embs, image_embed_dims, hidden_layer_factors):
        super().__init__()
        self.image_embed_dims = image_embed_dims
        self.param_shapes = param_shapes # `param_shapes = [[D_out, D_in], [D_out]]`
        self.hidden_layer_factors = hidden_layer_factors

        self.cond_embs = nn.Embedding(num_cond_embs, cond_emb_dim)
        self.shape_embs = nn.Embedding(len(image_embed_dims), cond_emb_dim)

        self.to_weight = MlpMapper(
            cond_emb_dim,
            [], #[f * cond_emb_dim for f in self.hidden_layer_factors], 
            param_shapes[0][0] * param_shapes[0][1]
        )
        self.to_bias = MlpMapper(
            cond_emb_dim, 
            [], #[f * cond_emb_dim for f in self.hidden_layer_factors], 
            param_shapes[1][0]
        )

    def compute_loss(self, logit_scale, image_features, text_features):
        logit_scale = logit_scale.exp().to(image_features.device)
        
        batch_size = image_features.shape[0]
        num_mappers = text_features.shape[0]

        labels = torch.arange(batch_size, dtype=torch.long).to(image_features.device).unsqueeze(0)
        labels = labels.repeat((num_mappers, 1))
        
        image_features = torch.permute(image_features, (1, 0, 2))
        logits1 = logit_scale * torch.einsum("nbd,ncd->nbc", image_features, text_features)

        preds = [logits1[i, :, :].argmax(dim=-1) for i in range(num_mappers)]
        corrects = [(preds[i] == labels[i, :]).sum().item() for i in range(num_mappers)]

        logits2 = logit_scale * torch.einsum("nbd,ncd->nbc", text_features, image_features)

        loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels))/2
        return loss.mean(), corrects


    def map_features(self, weights, biases, features):
        batch_size = features.shape[0]
        x = torch.einsum("nit,bt->nbi", weights, features)
        x = x + biases.unsqueeze(1).repeat((1, batch_size, 1))
        return x
    
    def forward(self, cond_id, image_embed_dim):
        if type(cond_id) != list:
            cond_id = [cond_id]

        cond_id = torch.tensor(cond_id).long().to(self.cond_embs.weight.device) 
        num_conds = len(cond_id)
        cond_emb = self.cond_embs(cond_id) # shape: [num_conds, cond_emb_dim]
        if num_conds == 1:
            cond_emb = cond_emb.unsqueeze(0)

        shape_id = torch.tensor([self.image_embed_dims.index(image_embed_dim)]).long().to(self.cond_embs.weight.device)
        shape_emb = self.shape_embs(shape_id) # shape: [1, cond_emb_dim]
        shape_emb = shape_emb.repeat((num_conds, 1)) # shape: [num_conds, cond_emb_dim]

        final_cond_emb = cond_emb + shape_emb

        # predict mappers
        pred_weight = self.to_weight(final_cond_emb) # shape [num_conds, flattened_weight_dim]
        pred_weight = pred_weight.view((num_conds, self.param_shapes[0][0], self.param_shapes[0][1]))

        pred_bias = self.to_bias(final_cond_emb)
        pred_bias = pred_bias.view((num_conds, self.param_shapes[0][0]))

        return pred_weight[:, :image_embed_dim, :], pred_bias[:, :image_embed_dim]


def main():
    device = "cuda"
    param_shapes = [[1024, 768], [1024]]
    model = MultiMapperHypernet(
        param_shapes, cond_emb_dim=8, num_cond_embs=8,
        image_embed_dims=[384,768], hidden_layer_factors=[4, 16]
    ).to(device)

    epoch = int(sys.argv[1])
    ckpt_path = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/multi_mapper_test_0_384-768/seed_0/ckpt_{epoch}.pt"
    ckpt = torch.load(ckpt_path)["model"]
    model.load_state_dict(ckpt)
    model.eval()

    weights, biases = model(cond_id=[0], image_embed_dim=768)
    configs = model_configs.ID_experiment_configs["id_vitr"]
    image_encoder = configs["image_encoders"][0]

    model = CustomVLM(image_encoder, "sentence-t5-base")
    model.mapper = MlpMapper(768, [], 768).to(device)

    model.mapper.layers[0].weight.data = weights[0]
    model.mapper.layers[0].bias.data = biases[0]

    cfg = data_configs.image_caption_dataset_configs["mscoco_val"]
    cfg["feature_dataset"] = "mscoco_val"
    cfg["transform"] = model.image_encoder.transform

    dataset = ImageCaptionDataset(cfg)
    loader = DataLoader(dataset, batch_size=1024, pin_memory=True, collate_fn=dataset.collate_fn, shuffle=False)

    separate_logs = coco_captions_eval(model, loader, progress_bar=True, using_clip=False)
    print(separate_logs)


if __name__ == "__main__":
    main()
 