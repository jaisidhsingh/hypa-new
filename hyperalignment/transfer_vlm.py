import timm
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


class CustomVLM():
    def  __init__(self, image_encoder_name, text_encoder_name, image_embed_dim=1024, text_embed_dim=768):
        self.device = "cuda"
        self.image_encoder = ImageEncoder(image_encoder_name)
        self.image_encoder.model = self.image_encoder.model.to(self.device)
        self.text_encoder = TextEncoder(text_encoder_name)
        self.text_encoder.model = self.text_encoder.model.to(self.device)
        self.mapper = nn.Linear(text_embed_dim, image_embed_dim)

    def encode_image(self, x):
        return self.image_encoder.encode_image(x)

    def encode_text(self, x):
        x = self.text_encoder.encode_text(x)
        x = x.to(self.device)
        return self.mapper(x)

    def encode_text_unmapped(self, x):
        x = self.text_encoder.encode_text(x)
        return x
    
    def load_mapper(self, ckpt_path):
        [w, b] = torch.load(ckpt_path)
        self.mapper.weight.data = w.cpu()
        self.mapper.bias.data = b.cpu()
