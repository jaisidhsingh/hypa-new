import os
import torch
import torchvision.datasets as torch_datasets
from torch.utils.data import Dataset
import random
from PIL import Image
import json


class CC3M300k(Dataset):
    def __init__(self, preprocessed_data_path, transform=None, path_appendage=None):
        data = torch.load(preprocessed_data_path)["test"]
        self.image_paths = data["image_paths"]

        if path_appendage is not None:
            self.image_paths = [path.replace("/workspace/datasets/cc3m/cc3m_subset_images_extracted_final", path_appendage) for path in self.image_paths]

        self.captions = [item["json"]["caption"] for item in data["labels"]]
        self.transform = transform
        del data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        # caption = self.annotations[idx]["json"]["caption"]
        caption = self.captions[idx]

        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)

        return image, [caption]


class CC3M595k(Dataset):
    def __init__(self, preprocessed_data_path, transform=None, path_appendage=None):
        with open(preprocessed_data_path) as f:
            data = json.load(f)

        self.image_paths = [item["image"] for item in data]
        self.captions = [item["blip_caption"] for item in data]
        self.image_paths = [os.path.join(path_appendage, path) for path in self.image_paths]

        self.transform = transform
        del data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        caption = self.captions[idx]

        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)

        return image, [caption]


class ImageCaptionDataset(Dataset):
    def __init__(self, kwargs):
        self.dataset_helper = None
        self.transform = kwargs["transform"]
        self.dataset_name = kwargs["feature_dataset"]

        if "mscoco" in kwargs["feature_dataset"]:
            kwargs.pop("feature_dataset")
            self.dataset_helper = torch_datasets.CocoCaptions(**kwargs)
        elif kwargs["feature_dataset"] == "cc3m300k":
            kwargs.pop("feature_dataset")
            self.dataset_helper = CC3M300k(**kwargs)
        elif kwargs["feature_dataset"] == "cc3m595k":
            kwargs.pop("feature_dataset")
            self.dataset_helper = CC3M595k(**kwargs)

    def __len__(self):
        return len(self.dataset_helper)

    def __getitem__(self, idx):
        image, captions = self.dataset_helper[idx]
        # if "mscoco" not in self.dataset_name:
            # captions = [captions[0]]
        # else:
            # captions = captions[:5]
        return image, captions

    def collate_fn(self, batch):
        B = len(batch)
        images = [None for _ in range(B)]

        if type(batch[0][0]) == torch.Tensor:
            images = [item[0].unsqueeze(0) for item in batch]
            [c, h, w] = images[0].shape[-3:]
            images = torch.cat(images, dim=0).view(B, c, h, w) ## ensuring shape is correct

        elif type(batch[0][0]) == Image:
            images = [item[0] for item in batch]

        if len(batch[0][1]) == 1:
            captions = [item[1][0] for item in batch]
        else:
            captions = [item[1] for item in batch]
        return images, captions
