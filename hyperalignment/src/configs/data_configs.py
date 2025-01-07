from types import SimpleNamespace


data_configs = SimpleNamespace(**{})
data_configs.STORE = "/home/mila/s/sparsha.mishra/scratch"

# data_configs.LOGS = "/home/mila/s/sparsha.mishra/projects/Hyper-Alignment/logs"
data_configs.LOGS = "./logs"

data_configs.embedding_store_root = f"{data_configs.STORE}/hyperalignment/results"

data_configs.image_caption_dataset_configs = {
	"mscoco_train": {
		"root": f"{data_configs.STORE}/coco_torchvision/extract/train2017",
		"annFile": f"{data_configs.STORE}/coco_torchvision/extract/annotations/captions_train2017.json"
	},
	"mscoco_val": {
		"root": f"{data_configs.STORE}/coco_torchvision/extract/val2017",
		"annFile": f"{data_configs.STORE}/coco_torchvision/captions_val2017.json"
	},
	"cc3m300k": {
		"preprocessed_data_path": f"{data_configs.STORE}/cc3m300k/preprocessed_data.pt",
	},
    "cc3m595k": {
		"preprocessed_data_path": f"{data_configs.STORE}/LLaVA-CC3M-Pretrain-595K/metadata.json",
        "caption_type": "raw"
	},
}

data_configs.image_classification_dataset_configs = {
    "cifar10": {"root": f"{data_configs.STORE}/cifar10_torchvision", "train": False, "download": False},
    "imagenet": {"root": f"{data_configs.STORE}/imagenet/val_torchvision/val"}
}

# This is a function to be called, i.e., `config = data_configs.joint_embedding_dataset_configs(args)`
# Accessed in `hyperalignment/experiments/learn_joint_mapping.py`
data_configs.joint_embedding_dataset_configs = lambda args: {
	"image_embeddings_path": f"{args.results_folder}/image_embeddings/{args.feature_dataset}/dim_{args.image_embed_dim}.pt",
	"text_embeddings_path": f"{args.results_folder}/text_embeddings/{args.feature_dataset}/dim_{args.text_embed_dim}.pt",
	"image_embed_dim": args.image_embed_dim,
	"text_embed_dim": args.text_embed_dim,
	"chosen_image_encoders": args.chosen_image_encoders,
	"chosen_text_encoders": args.chosen_text_encoders,
	"feature_dataset": args.feature_dataset
}

# This is a function to be called, i.e., `config = data_configs.separate_embedding_dataset_configs(args)`
# Accessed in `hyperalignment/experiments/learn_separate_mapping.py`
data_configs.separate_embedding_dataset_configs = lambda args: {
	# "image_embeddings_path": f"{args.results_folder}/image_embeddings/{args.feature_dataset}/dim_{args.image_embed_dim}.pt",
	# "text_embeddings_path": f"{args.results_folder}/text_embeddings/{args.feature_dataset}/dim_{args.text_embed_dim}.pt",
    "image_embeddings_path": f"{args.results_folder}/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_{args.image_embed_dim}/{args.image_encoder}/memmap.npy",
    "text_embeddings_path": f"{args.results_folder}/text_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie/dim_{args.text_embed_dim}/{args.text_encoder}/memmap.npy",
	"image_embed_dim": args.image_embed_dim,
	"text_embed_dim": args.text_embed_dim,
	"image_encoder": args.image_encoder,
	"text_encoder": args.text_encoder,
    "num_samples": 595375,
	"feature_dataset": args.feature_dataset
}

data_configs.multi_embedding_dataset_configs = {
    "cc3m595k": {
		"image_data_folder": f"{data_configs.STORE}/hyperalignment/results/image_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie",
		"image_encoder_data": None,
		"text_data_folder": f"{data_configs.STORE}/hyperalignment/results/text_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie",
		"text_encoder": "sentence-t5-base",
		"text_embed_dim": 768,
		"num_samples": 595375
	},

    "cc3m595k_8-4": {
		"image_data_folder": f"{data_configs.STORE}/hyperalignment/results/image_embeddings/cc3m595k_multi_mapper",
		"image_encoder_data": None,
		"text_data_folder": f"{data_configs.STORE}/hyperalignment/results/text_embeddings/multi_mapper/cc3m595k_multi_mapper_30_ie",
		"text_encoder": "sentence-t5-base",
		"text_embed_dim": 768,
		"num_samples": 595375
	}
}
