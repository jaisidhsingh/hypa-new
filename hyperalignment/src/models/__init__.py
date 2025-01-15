import timm
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .param_decoders import *


class ImageEncoder(nn.Module):
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device

        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**self.config, is_training=False)

    def encode_image(self, image):
        try:
            x = image.shape
        except:
            image = self.transform(image)

        image_features = self.model(image)
        return F.normalize(image_features, dim=-1)

    def forward(self, image):
        return self.encode_image(image)


class TextEncoder():
	def __init__(self, model_name, device="cuda"):
		self.model_name = model_name
		self.device = device

		self.model = SentenceTransformer(model_name).to(device)
		self.model.eval()

	def encode_text(self, sentences):
		text_features = self.model.encode(sentences)
		text_features = torch.from_numpy(text_features)
		return F.normalize(text_features, dim=-1)


class CustomVLM():
    def  __init__(self, image_encoder_name, text_encoder_name):
        self.device = "cuda"
        self.image_encoder = ImageEncoder(image_encoder_name)
        self.image_encoder.model = self.image_encoder.model.to(self.device)
        self.text_encoder = TextEncoder(text_encoder_name)
        self.text_encoder.model = self.text_encoder.model.to(self.device)
        self.mapper = None

    def encode_image(self, x):
        return self.image_encoder.encode_image(x)

    def encode_text(self, x):
        x = self.text_encoder.encode_text(x)
        x = x.to(self.device)
        return self.mapper(x)

    def encode_text_unmapped(self, x):
        x = self.text_encoder.encode_text(x)
        return x


class ConditionalHyperNetwork(nn.Module):
    def __init__(self, param_shapes, cond_emb_dim, num_cond_embs, image_embed_dims, kwargs):
        super().__init__()
        self.image_embed_dims = image_embed_dims
        self.param_shapes = param_shapes # `param_shapes = [[D_out, D_in], [D_out]]`

        self.num_cond_embs = num_cond_embs
        self.cond_embs = nn.Embedding(num_cond_embs, cond_emb_dim)
        self.shape_embs = nn.Embedding(len(image_embed_dims), cond_emb_dim)

        # self.in_proj = nn.Linear(cond_emb_dim, 4*cond_emb_dim)
        # self.attn = nn.TransformerEncoderLayer(4*cond_emb_dim, 8, batch_first=True)

        self.cond_dim = cond_emb_dim

        self.decoder = None
        
        if kwargs["decoder_type"] == "mlp":
            self.decoder = MlpDecoder(param_shapes[0], cond_emb_dim, kwargs["hidden_layer_factors"])
        
        elif kwargs["decoder_type"] == "attention":
            self.decoder = AttentionDecoder(param_shapes[0], cond_emb_dim, kwargs["num_layers"], kwargs["num_heads"], kwargs["expansion_factor"])

        assert self.decoder is not None, "Decoder type not recognized."


    def lookup_embedding_table(self, cond_id): 
        assert type(cond_id) in [list, int], "Conditional input is of the wrong type."

        if type(cond_id) != list and type(cond_id) == int:
            cond_id = [cond_id]

        cond_id = torch.tensor(cond_id).long().to(self.cond_embs.weight.device) 
        num_conds = len(cond_id)
        cond_emb = self.cond_embs(cond_id) # shape: [num_conds, cond_emb_dim]

        if num_conds == 1:
            cond_emb = cond_emb.unsqueeze(0)
        
        return cond_emb

    
    def forward(self, cond_id, image_embed_dim, normalize_output=False, nolookup=False):
        if nolookup == False:
            assert type(cond_id) in [list, int], "Conditional input is of the wrong type."

            if type(cond_id) != list and type(cond_id) == int:
                cond_id = [cond_id]

            cond_id = torch.tensor(cond_id).long().to(self.cond_embs.weight.device) 
            num_conds = len(cond_id)
            cond_emb = self.cond_embs(cond_id) # shape: [num_conds, cond_emb_dim]
            if num_conds == 1:
                cond_emb = cond_emb.unsqueeze(0)
        
        else:
            assert type(cond_id) in [torch.Tensor, torch.nn.Parameter], "Conditional input is of the wrong type."
            num_conds = cond_id.shape[0]
            cond_emb = cond_id

        shape_id = torch.tensor([self.image_embed_dims.index(image_embed_dim)]).long().to(self.cond_embs.weight.device)
        shape_emb = self.shape_embs(shape_id) # shape: [1, cond_emb_dim]
        shape_emb = shape_emb.repeat((num_conds, 1)) # shape: [num_conds, cond_emb_dim]

        final_cond_emb = cond_emb + shape_emb
        # final_cond_emb = self.in_proj(final_cond_emb)
        # final_cond_emb = self.attn(final_cond_emb.unsqueeze(0)).squeeze(0)

        pred_weight, pred_bias = self.decoder(final_cond_emb)
        pred_weight = pred_weight[:, :image_embed_dim, :]
        pred_bias = pred_bias[:, :image_embed_dim]

        if normalize_output:
            pred_weight = pred_weight * (1 / pred_weight[0].numel()) ** 0.5
            pred_bias = pred_bias * (1 / pred_bias[0].numel()) ** 0.5
        
        return pred_weight, pred_bias


    def map_features(self, weights, biases, features):
        batch_size = features.shape[0]
        x = torch.einsum("nit,bt->nbi", weights, features)
        x = x + biases.unsqueeze(1).repeat((1, batch_size, 1))
        x = x / x.norm(dim=-1, keepdim=True)
        return x


    def compute_loss(self, logit_scale, image_features, text_features, emb_loss=False):
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

        if emb_loss:
            self.cond_embs.weight.data = self.cond_embs.weight.data / self.cond_embs.weight.data.norm(dim=-1, keepdim=True)

            emb_sim = self.cond_embs.weight @ self.cond_embs.weight.T
            emb_labels = torch.arange(self.num_cond_embs, dtype=torch.long).to(self.cond_embs.weight.data)
            emb_l = F.cross_entropy(emb_sim.float(), emb_labels.long())
        else:
            emb_l = 0
        
        loss = loss.mean() + emb_l

        return loss, corrects
