import timm
import numpy as np
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


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


class MlpMapper(nn.Module):
	def __init__(self, input_dim: int, intermediate_dims: List, output_dim: int, use_bias: bool = True, logit_scale: float = 100.0):
		super().__init__()
		self.input_dim = input_dim
		self.intermediate_dims = intermediate_dims # list of ints
		self.output_dim = output_dim
		self.num_layers = len(intermediate_dims) + 1

		self.layers = []
		current_dim = input_dim
		next_dims = intermediate_dims + [output_dim]

		if logit_scale < 0:
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		else:
			self.logit_scale = torch.tensor(np.log(logit_scale))

		for i in range(self.num_layers):
			self.layers.append(nn.Linear(current_dim, next_dims[i], bias=use_bias))
			current_dim = next_dims[i]

			if i != self.num_layers - 1:
				self.layers.append(nn.GELU())

		self.layers = nn.Sequential(*self.layers)

	def forward(self, x):
		x = self.layers(x)
		return F.normalize(x, dim=-1)


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


class MultiMapperHypernet(nn.Module):
    def __init__(self, param_shapes, cond_emb_dim, num_cond_embs, image_embed_dims, hidden_layer_factors, rescale_factor=10):
        super().__init__()
        self.image_embed_dims = image_embed_dims
        self.param_shapes = param_shapes # `param_shapes = [[D_out, D_in], [D_out]]`
        self.hidden_layer_factors = hidden_layer_factors

        self.cond_embs = nn.Embedding(num_cond_embs, cond_emb_dim)
        self.shape_embs = nn.Embedding(len(image_embed_dims), cond_emb_dim)

        self.to_weight = MlpMapper(
            cond_emb_dim,
            [f * cond_emb_dim for f in self.hidden_layer_factors], 
            param_shapes[0][0] * param_shapes[0][1]
        )
        self.to_bias = MlpMapper(
            cond_emb_dim, 
            [f * cond_emb_dim for f in self.hidden_layer_factors], 
            param_shapes[1][0]
        )
        if rescale_factor != 0.0:
            self.rescale_weight_prediction_params(rescale_factor)
        
        self.rescale_factor = rescale_factor     
    
    def rescale_weight_prediction_params(self, rescale_factor):
        # rescale the `weight` tensor data by `scale_factor` and set the bias to 0
        num_layers = self.to_weight.num_layers
        for i in [-1]:
            if hasattr(self.to_weight.layers[i], "weight") and hasattr(self.to_weight.layers[i], "bias"):
                self.to_weight.layers[i].bias.data.fill_(0.)
                self.to_weight.layers[i].weight.data /= rescale_factor

            if hasattr(self.to_bias.layers[i], "weight") and hasattr(self.to_bias.layers[i], "bias"):
                self.to_bias.layers[i].weight.data /= rescale_factor
                self.to_bias.layers[i].bias.data.fill_(0.)

        print("Rescaled parameters of `self.to_weight` and `self.to_bias`.")


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
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def forward(self, cond_id, image_embed_dim, normalize_output=False):
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

        pred_weight = pred_weight[:, :image_embed_dim, :]
        pred_bias = pred_bias[:, :image_embed_dim]

        if normalize_output:
            pred_weight = pred_weight * (1 / pred_weight[0].numel()) ** 0.5
            pred_bias = pred_bias * (1 / pred_bias[0].numel()) ** 0.5
        
        return pred_weight, pred_bias


class OuterProdParamDecoder(nn.Module):
    def __init__(self, input_dim, output_dims):
        super().__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims # (dest_dim, src_dim)

        self.decoder = nn.ModuleList([
            nn.Linear(input_dim, output_dims[0]),
            nn.Linear(input_dim, output_dims[1])
        ])
    
    def forward(self, x):
        out1, out2 = self.decoder[0](x), self.decoder[1](x)
        return torch.einsum("bi,bj->bij", out1, out2)


class OuterProdHypNet(nn.Module):
    def __init__(self, param_shapes, cond_emb_dim, num_cond_embs, image_embed_dims, hidden_layer_factors, rescale_factor=10):
        super().__init__()
        print("Initialising hypernetwork which uses outer product.")
        self.image_embed_dims = image_embed_dims
        self.param_shapes = param_shapes # `param_shapes = [[D_out, D_in], [D_out]]`
        self.hidden_layer_factors = hidden_layer_factors

        self.cond_embs = nn.Embedding(num_cond_embs, cond_emb_dim)
        self.shape_embs = nn.Embedding(len(image_embed_dims), cond_emb_dim)

        self.to_weight = OuterProdParamDecoder(cond_emb_dim, [param_shapes[0][0], param_shapes[0][1]])
        self.to_bias = OuterProdParamDecoder(cond_emb_dim, [param_shapes[1][0], 1])

    def forward(self, cond_id, image_embed_dim, normalize_output=False):
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

        pred_weight = self.to_weight(final_cond_emb)
        pred_bias = self.to_bias(final_cond_emb).squeeze(-1)

        pred_weight = pred_weight[:, :image_embed_dim, :]
        pred_bias = pred_bias[:, :image_embed_dim]

        if normalize_output:
            pred_weight = pred_weight * (1 / pred_weight[0].numel()) ** 0.5
            pred_bias = pred_bias * (1 / pred_bias[0].numel()) ** 0.5
        
        return pred_weight, pred_bias


    def rescale_weight_prediction_params(self, rescale_factor):
        # rescale the `weight` tensor data by `scale_factor` and set the bias to 0
        num_layers = self.to_weight.num_layers
        for i in [-1]:
            if hasattr(self.to_weight.layers[i], "weight") and hasattr(self.to_weight.layers[i], "bias"):
                self.to_weight.layers[i].bias.data.fill_(0.)
                self.to_weight.layers[i].weight.data /= rescale_factor

            if hasattr(self.to_bias.layers[i], "weight") and hasattr(self.to_bias.layers[i], "bias"):
                self.to_bias.layers[i].weight.data /= rescale_factor
                self.to_bias.layers[i].bias.data.fill_(0.)

        print("Rescaled parameters of `self.to_weight` and `self.to_bias`.")


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
        x = x / x.norm(dim=-1, keepdim=True)
        return x
