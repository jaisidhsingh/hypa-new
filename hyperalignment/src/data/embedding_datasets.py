import os
import numpy as np
import torch
import random
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset


class JointEmbeddings(Dataset):
    def __init__(self, data_config):
        self.image_embeddings = torch.load(data_config["image_embeddings_path"])[data_config["image_embed_dim"]]
        self.text_embeddings = torch.load(data_config["text_embeddings_path"])[data_config["text_embed_dim"]]
        
        all_image_encoders = list(self.image_embeddings.keys())
        all_text_encoders = list(self.text_embeddings.keys())

        chosen_image_encoders = [int(x) for x in data_config["chosen_image_encoders"].split(",")]
        chosen_image_encoders = [all_image_encoders[idx] for idx in chosen_image_encoders]

        chosen_text_encoders = [int(x) for x in data_config["chosen_text_encoders"].split(",")]
        chosen_text_encoders = [all_text_encoders[jdx] for jdx in chosen_text_encoders]
        
        unwanted_image_encoders = list(set(all_image_encoders) - set(chosen_image_encoders))
        unwanted_text_encoder = list(set(all_text_encoders) - set(chosen_text_encoders))

        for ie in unwanted_image_encoders:
            self.image_embeddings.pop(ie)
        
        for te in unwanted_text_encoder:
            self.text_embeddings.pop(te)
        
        self.image_keys = list(self.image_embeddings.keys())
        # self.image_key_indices = [i for i in range(len(self.image_keys))]
        
        self.text_keys = list(self.text_embeddings.keys())
        # self.text_key_indices = [i for i in range(len(self.text_keys))]
        
        self.dataset_length = self.image_embeddings[self.image_keys[0]].shape[0]
        self.image_embed_dim = data_config["image_embed_dim"]
        self.text_embed_dim = data_config["text_embed_dim"]
        self.feature_dataset = data_config["feature_dataset"]
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        all_image_embeddings = torch.cat([
            self.image_embeddings[k][idx].cpu().unsqueeze(0) for k in self.image_keys
        ], dim=0).view(len(self.image_keys), self.image_embed_dim)
        
        all_text_embeddings = torch.cat([
            self.text_embeddings[k][idx].cpu().unsqueeze(0) for k in self.text_keys
        ], dim=0).view(len(self.text_keys), self.text_embed_dim)
        
        return all_image_embeddings, all_text_embeddings


class SeparateEmbeddings(Dataset):
    def __init__(self, data_config, split, args):
        # self.image_embeddings = torch.load(data_config["image_embeddings_path"])[data_config["image_embed_dim"]][data_config["image_encoder"]].cpu()
        # self.text_embeddings = torch.load(data_config["text_embeddings_path"])[data_config["text_embed_dim"]][data_config["text_encoder"]].cpu()
        self.image_embeddings = np.memmap(data_config["image_embeddings_path"], dtype="float32", mode="r", shape=(data_config["num_samples"], data_config["image_embed_dim"]))
        self.text_embeddings = np.memmap(data_config["text_embeddings_path"], dtype="float32", mode="r", shape=(data_config["num_samples"], data_config["text_embed_dim"]))

        self.split = split
        self.split_ratio = args.train_val_split_ratio
        self.data_config = data_config

        if split == "train":
            self.num_samples = round(self.split_ratio * data_config["num_samples"])
        elif split == "val":
            self.num_samples = data_config["num_samples"] - round(self.split_ratio * data_config["num_samples"])

        # self.indices = [i for i in range(data_config["num_samples"])]
        # random.seed(args.random_seed)
        # random.shuffle(self.indices)
        
        self.image_embed_dim = data_config["image_embed_dim"]
        self.text_embed_dim = data_config["text_embed_dim"]
        self.feature_dataset = data_config["feature_dataset"]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # image_embedding = self.image_embeddings[idx].view(1, self.image_embed_dim)
        # text_embedding = self.text_embeddings[idx].view(1, self.text_embed_dim)
        
        offset = 0 if self.split == "train" else round(self.split_ratio * self.data_config["num_samples"])
        index = offset + idx

        image_embedding = np.array(deepcopy(self.image_embeddings[index, :])).astype(np.float32)
        text_embedding = np.array(deepcopy(self.text_embeddings[index, :])).astype(np.float32)

        return torch.from_numpy(image_embedding), torch.from_numpy(text_embedding)

        # image_embedding = torch.from_numpy(self.image_embeddings[idx]).view(self.image_embed_dim)
        # text_embedding = torch.from_numpy(self.text_embeddings[idx]).view(self.text_embed_dim)
        # return image_embedding, text_embedding


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
        return self.num_samples

    def get_minibatch(self, batch_indices, sampled_encoder_indices, encoder_dims):
        # first get the encoder names from indices
        sampled_encoders = [self.image_encoders[i] for i in sampled_encoder_indices]
        # print(sampled_encoders)
    
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
        # text_embeddings = torch.from_numpy(np.load(text_embeddings_path))[start:end, :]
        text_embeddings_memmap = np.memmap(text_embeddings_path, dtype="float32", mode="r", shape=(self.num_samples, self.text_embed_dim))
        text_embeddings = torch.from_numpy(np.array(text_embeddings_memmap[start:end, :]))
        assert text_embeddings.shape == torch.Size([end-start, self.text_embed_dim]), "Text embeddings prepared incorrectly!"

        return image_embeddings, text_embeddings
