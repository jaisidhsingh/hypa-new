import os
import numpy as np
import torch
import random
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset


class SeparateEmbeddings(Dataset):
    def __init__(self, data_config, split, args):
        if os.path.exists(data_config["image_embeddings_path"]):
            self.image_embeddings = np.memmap(data_config["image_embeddings_path"], dtype="float32", mode="r", shape=(data_config["num_samples"], data_config["image_embed_dim"]))
        else:
            self.image_embeddings = np.load(data_config["image_embeddings_path"].replace("memmap", "embeddings"), allow_pickle=True)
        self.text_embeddings = np.memmap(data_config["text_embeddings_path"], dtype="float32", mode="r", shape=(data_config["num_samples"], data_config["text_embed_dim"]))

        self.data_config = data_config
        self.image_embed_dim = data_config["image_embed_dim"]
        self.text_embed_dim = data_config["text_embed_dim"]
        self.feature_dataset = data_config["feature_dataset"]
        self.data_scaling = args.data_scaling
        
    def __len__(self):
        return round(558128 * self.data_scaling) #self.data_config["num_samples"]
    
    def __getitem__(self, idx):
        image_embedding = np.array(deepcopy(self.image_embeddings[idx, :])).astype(np.float32)
        text_embedding = np.array(deepcopy(self.text_embeddings[idx, :])).astype(np.float32)
        return torch.from_numpy(image_embedding), torch.from_numpy(text_embedding)


class ImageEmbeddings(Dataset):
    def __init__(self, data_config, split=None, args=None):
        if os.path.exists(data_config["image_embeddings_path"]):
            self.image_embeddings = np.memmap(data_config["image_embeddings_path"], dtype="float32", mode="r", shape=(data_config["num_samples"], data_config["image_embed_dim"]))
        else:
            self.image_embeddings = np.load(data_config["image_embeddings_path"].replace("memmap", "embeddings"), allow_pickle=True)

        self.data_config = data_config
        self.image_embed_dim = data_config["image_embed_dim"]
        self.feature_dataset = data_config["feature_dataset"]
        
    def __len__(self):
        return 558128 #self.data_config["num_samples"]
    
    def __getitem__(self, idx):
        image_embedding = np.array(deepcopy(self.image_embeddings[idx, :])).astype(np.float32)
        return torch.from_numpy(image_embedding)


class TextEmbeddings(Dataset):
    def __init__(self, data_config, split=None, args=None):
        self.text_embeddings = np.memmap(data_config["text_embeddings_path"], dtype="float32", mode="r", shape=(data_config["num_samples"], data_config["text_embed_dim"]))

        self.data_config = data_config
        self.text_embed_dim = data_config["text_embed_dim"]
        self.feature_dataset = data_config["feature_dataset"]
        
    def __len__(self):
        return 558128 #self.data_config["num_samples"]
    
    def __getitem__(self, idx):
        text_embedding = np.array(deepcopy(self.text_embeddings[idx, :])).astype(np.float32)
        return torch.from_numpy(text_embedding)


class MultiMapperEmbeddings(Dataset):
    def __init__(self, config):
        # image encoders metadata
        self.image_data_folder = config["image_data_folder"]
        self.image_encoder_data = config["image_encoder_data"]
        self.image_encoders = [y for x in self.image_encoder_data.values() for y in x]
        self.image_embed_dims = [k for k in self.image_encoder_data.keys()]
        self.num_image_encoders = len(self.image_encoders)

        # text encoder metadata
        self.text_data_folder = config["text_data_folder"]
        self.text_encoder = config["text_encoder"]
        self.text_embed_dim = config["text_embed_dim"]
        self.num_text_encoders = 1

        # misc metadata
        self.num_samples = config["num_samples"]
        self.args = config["args"]

        # make `np.memmaps` of all the data
        # first for image embeddings
        print("Checking and preparing memmaps...")
        for dim in self.image_embed_dims:
            for encoder_name in self.image_encoder_data[dim]:
                if not os.path.exists(os.path.join(self.image_data_folder, f"dim_{dim}", encoder_name, "memmap.npy")):
                    self.make_memmap(dim, encoder_name)

        # then for image embeddings 
        if not os.path.exists(os.path.join(self.text_data_folder, f"dim_{self.text_embed_dim}", self.text_encoder, "memmap.npy")):
            self.make_memmap(self.text_embed_dim, self.text_encoder, mode="text")

        print("Memmaps prepared.")

    def make_memmap(self, dim, encoder_name, mode="image"):
        if mode == "image":
            folder = os.path.join(self.image_data_folder, f"dim_{dim}", encoder_name)
        elif mode == "text":
            folder = os.path.join(self.text_data_folder, f"dim_{self.text_embed_dim}", self.text_encoder)

        emb_path = os.path.join(folder, "embeddings.npy")
        embeddings = np.load(emb_path)

        memmap_path = os.path.join(folder, "memmap.npy")
        mmap = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(self.num_samples, dim))

        mmap[:, :] = embeddings[:, :]
        assert mmap[0, :3].all() == embeddings[0, :3].all(), "Mismatch in memmap!"
        mmap.flush()
        
        re_mmap = np.memmap(memmap_path, dtype="float32", mode="r", shape=(self.num_samples, dim))
        assert re_mmap[0, :3].all() == embeddings[0, :3].all(), "Mismatch in memmap!"

    def __len__(self):
        return round(558128 * self.args.dataset_scale)

    def get_minibatch(self, batch_indices, sampled_encoder_indices, encoder_dims):
        # first get the encoder names from indices
        sampled_encoders = [self.image_encoders[i] for i in sampled_encoder_indices]
        chosen_dim = encoder_dims[0]
        assert chosen_dim in self.image_embed_dims, "Error in dimension sampling!"

        # remember to not shuffle the data
        start, end = batch_indices[0], batch_indices[-1]+1

        # get image embeddings
        image_embeddings_paths = [
            os.path.join(
                self.image_data_folder, 
                f"dim_{chosen_dim}", 
                encoder, 
                "memmap.npy"
            ) for encoder in sampled_encoders
        ]

        image_embeddings_memmaps = [np.memmap(path, dtype="float32", mode="r", shape=(self.num_samples, chosen_dim)) for path in image_embeddings_paths]
        image_embeddings = [np.array(memmap[start:end, :]) for memmap in image_embeddings_memmaps]
        image_embeddings = np.concatenate(
            [
                np.expand_dims(emb, axis=1) for emb in image_embeddings
            ], 
            axis=1
        )
        image_embeddings = torch.from_numpy(image_embeddings)
        assert image_embeddings.shape == torch.Size([end-start, len(sampled_encoders), chosen_dim]), "Image embeddings prepared incorrectly!"

        # get text embeddings
        text_embeddings_path = os.path.join(self.text_data_folder, f"dim_{self.text_embed_dim}", self.text_encoder, "memmap.npy")
        text_embeddings_memmap = np.memmap(text_embeddings_path, dtype="float32", mode="r", shape=(self.num_samples, self.text_embed_dim))
        text_embeddings = torch.from_numpy(np.array(text_embeddings_memmap[start:end, :]))
        assert text_embeddings.shape == torch.Size([end-start, self.text_embed_dim]), "Text embeddings prepared incorrectly!"

        return image_embeddings, text_embeddings


# class JointEmbeddings(Dataset):
#     def __init__(self, text_memmap_folder, image_memmap_folder, encoder_names, args):
#         self.image_memmap_paths = [os.path.join(image_memmap_folder, args.image_embed_dim, name) for name in encoder_names]
#         self.image_embedding_memmaps = [np.memmap(path, dtype=np.flot32, mode="r", shape=(595375, args.image_embed_dim)) for path in self.image_memmap_paths]
#         self.text_embedding_memmap = np.memmap(os.path.join(text_memmap_folder, args.text_embed_dim, args.text_encoder, "memmap.npy"), mode="r", dtype=np.float32, shape=(595375, args.text_embed_dim))
#         self.args = args
    
#     def __len__(self):
#         return 558128
    
#     def __getitem__(self, idx):
#         image_embeddings = [np.array(image_memmaps[idx]) for image_memmaps in self.image_embedding_memmaps]
#         image_embeddings = torch.from_numpy(np.stack(image_embeddings)).view(1, self.args.num_image_encoders, self.args.image_embed_dim)
#         text_embeddings = torch.from_numpy(np.array(self.text_embedding_memmap[idx]))
#         return image_embeddings, text_embeddings
