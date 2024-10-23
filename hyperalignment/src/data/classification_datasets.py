import os
from torch.utils.data import Dataset
from torchvision import datasets as torch_datasets


class ImageClassificationDataset(Dataset):
	def __init__(self, kwargs):
		self.helper_map = {
			"cifar10": torch_datasets.CIFAR10,
			"cifar100": torch_datasets.CIFAR100,
			"imagenet1k": torch_datasets.ImageFolder,
		}
		self.dataset_name = kwargs["feature_dataset"]
		kwargs.pop("feature_dataset")
		self.classes = None
		if "cifar" in self.dataset_name:
			kwargs["download"] = True
			kwargs["train"] = False
		self.dataset_helper = self.helper_map[self.dataset_name](**kwargs)
		class_names = self.get_class_names()

	def get_class_names(self):
		if "imagenet" in self.dataset_name:
			class_names = {}
			path = "/home/mila/s/sparsha.mishra/projects/hypa-new/hyperalignment/src/data/imagenet_class_mapping.txt"

			with open(path) as f:
				for line in f.readlines():
					entry = line.split(" ")
					key = entry[0]
					value = entry[-1]
					class_names[key] = value[:-1].lower().replace("_", " ")

			self.classes = [class_names[c] for c in self.dataset_helper.classes]
		else:
			self.classes = self.dataset_helper.classes
		
		return self.classes

	def __len__(self):
		return len(self.dataset_helper)

	def __getitem__(self, idx):
		(image, label) = self.dataset_helper[idx]
		return image, label


if __name__ == "__main__":
	kwargs = {
		"feature_dataset": "imagenet",
		"root": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
		"transform": None
	}
	dataset = ImageClassificationDataset(kwargs)
	classes = dataset.get_class_names()
	print(len(classes), len(dataset))
