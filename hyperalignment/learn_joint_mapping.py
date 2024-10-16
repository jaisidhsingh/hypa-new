import os
import json
import math
import torch
import warnings
import argparse
from tqdm import tqdm
from copy import deepcopy
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP, ChunkedHMLP
from torch.utils.data import DataLoader

from src.data import *
from src.models import *
from src.configs.data_configs import data_configs

from src.utils import get_hypnet_flops
from src.utils.weight_analysis import WeightAnalyzer

from src.training import JointTrainer
from src.training.schedulers import cosine_lr
from src.training.loss_functions import ClipLoss
warnings.simplefilter("ignore")


def apply_params_to_mlp(params, args):
    # initialize the mapper with the weights above
    model = MlpMapper(args.text_embed_dim, [], args.image_embed_dim, use_bias=args.use_bias, logit_scale=args.logit_scale)
    model = model.to(args.device)
    model.layers[0].weight.data = params[0]
    model.layers[0].bias.data = params[1]
    return model


def train_joint_mapper(args):
	# set the seed
	torch.manual_seed(args.random_seed)

	# load in dataset for training
	train_dataset_config = data_configs.joint_embedding_dataset_configs(args)
	train_dataset = JointEmbeddings(train_dataset_config)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=args.shuffle_data)
	print("Data loaded.")

	"""
	# load in dataset for training
	val_args = deepcopy(args)
	val_args.feature_dataset = args.val_dataset
	val_dataset_config = data_configs.joint_embedding_dataset_configs(val_args)
	val_dataset = JointEmbeddings(val_dataset_config)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=args.shuffle_data)
	print("Data loaded.")

	# load in dataset for testing
	test_args = deepcopy(args)
	test_args.feature_dataset = args.test_dataset
	test_dataset_config = data_configs.joint_embedding_dataset_configs(test_args)
	test_dataset = JointEmbeddings(test_dataset_config)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=args.shuffle_data)
	print("Data loaded.")
    """

	# initialize the linear layer mapper
	main_model = MLP(n_in=args.text_embed_dim, n_out=args.image_embed_dim, hidden_layers=[], no_weights=True).to(args.device)
	print(f"Number of params in mapper: {len(list(main_model.parameters()))}")

	hnet = None
	num_image_encoders = len(args.chosen_image_encoders.split(","))
	num_text_encoders = len(args.chosen_text_encoders.split(","))

	if args.hnet_type == "simple":
		# initialize the hyper-network which predicts the linear layer
		hnet = HMLP(main_model.param_shapes, uncond_in_size=0,
			cond_in_size=args.hnet_cond_emb_dim, layers=[],
			num_cond_embs=int(num_image_encoders * num_text_encoders)
		).to(args.device)

		# apply `hyperfan` to hyper-network weights [recommended]
		hnet.apply_hyperfan_init(mnet=main_model)
		hnet.train()

	elif args.hnet_type == "chunked":
		hnet = ChunkedHMLP(
			main_model.param_shapes, uncond_in_size=0,
			cond_in_size=args.hnet_cond_emb_dim, layers=[],
			cond_chunk_embs=args.num_chunks, chunk_size=int(args.image_embed_dim // args.num_chunks),
			num_cond_embs=int(num_image_encoders * num_text_encoders)
		).to(args.device)
		# hnet.apply_hyperfan_init(mnet=main_model)
		hnet.train()

	elif args.hnet_type == "lora":
		hnet = LoraHypnet(main_model.param_shapes, cond_emb_size=args.hnet_cond_emb_dim,
			num_cond_embs=int(num_image_encoders * num_text_encoders),
			rank=args.lora_rank).to(args.device)
		hnet.train()

	# count FLOPs taken
	total_flops = get_hypnet_flops(hnet, {"cond_id": 0}, include_backward=True)
	print(f"Trainable FLOPs for hyper-network (per mapping): {total_flops[0]} x 10^{math.log10(total_flops[1])}")

	# optimizer, loss, scheduler, scaler, logit_scale and log stores
	optimizer = torch.optim.AdamW(hnet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	criterion = ClipLoss(args)

	if args.scheduler != "off":
		total_steps = int(args.num_epochs * len(train_loader))
		scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)
	else:
		scheduler = None

	scaler = torch.cuda.amp.GradScaler()
	logs = {}
	logs[f"configs"] = {
		"train_dataset": args.feature_dataset,
		"batch_size": args.batch_size,
		"num_epochs": args.num_epochs,
		"steps_per_epoch": len(train_loader),
		"flops_per_epoch": total_flops[0] * total_flops[1]
	}
	weight_analysis_logs = {}

	# setup folder to save checkpoints in
	ckpt_save_folder = os.path.join(args.checkpoint_folder, args.experiment_type, "joint", args.experiment_name, f"seed_{args.random_seed}")
	os.makedirs(ckpt_save_folder, exist_ok=True)
	# setup folder to save logs in
	logs_save_folder = os.path.join(args.logs_folder, args.experiment_type, "joint", args.experiment_name, f"seed_{args.random_seed}")
	os.makedirs(logs_save_folder, exist_ok=True)

	trainer = JointTrainer(args)

	# training loop
	bar = tqdm(total=args.num_epochs)
	for epoch in range(args.num_epochs):
		train_logs = trainer.train_one_epoch(hnet, main_model, train_loader, criterion, optimizer, scheduler, scaler, epoch)
		# val_logs = trainer.val_one_epoch(hnet, main_model, val_loader, criterion)
		# test_logs = trainer.val_one_epoch(hnet, main_model, test_loader, criterion)

		logs[f"epoch_{epoch+1}"] = {"train": train_logs} #, "val": val_logs, "test": test_logs}
		to_log = {
			# "val_loss": val_logs["avg_loss"],
			# "val_acc": val_logs["accuracies"]
		}
		to_desc = f"train_loss: {train_logs['avg_loss']}  train_acc: {train_logs['accuracies']}"
		bar.set_postfix(to_log)
		bar.set_description(to_desc)

		# make sure we save some epochs
		if (epoch+1) in [1, 5, 10, 20] and args.saving:
			N = int(num_image_encoders * num_text_encoders)
			params = hnet(cond_id=[i for i in range(N)])
			dump = {
				"args": args,
				"mapper_params": params,
				"optimizer": optimizer.state_dict(),
				"logs": logs
			}
			torch.save(dump, os.path.join(ckpt_save_folder, f"ckpt_{epoch+1}.pt"))
			tqdm.write(f"Checkpoint saved at epoch {epoch+1}.")

			weight_analysis_logs[f"epoch_{epoch}"] = {}
			for ii, param in enumerate(params):
				analysis_model = apply_params_to_mlp(param, args)
				analyser = WeightAnalyzer(analysis_model)

				weight_analysis_logs[f"epoch_{epoch}"][str(ii)] = {
					"esd_array": analyser.esd.tolist(),
					"esd_hist_x": analyser.esd_hist_x.tolist(),
					"esd_hist_y": analyser.esd_hist_y.tolist(),
					"alpha": analyser.alpha,
					"lambda_xmin": analyser.lambda_xmin,
					"lambda_xmax": analyser.lambda_xmax,
					"summary": analyser.summary
				}

		bar.update(1)

	# close the bar after training
	bar.close()
	if args.saving:
		# save logs
		logs.update({"args": args.__dict__})
		with open(os.path.join(logs_save_folder, "train_logs.json"), "w") as f:
			json.dump(logs, f)

		# save weight analysis
		with open(os.path.join(logs_save_folder, "pre_ft_weight_analysis_logs.json"), "w") as f:
			json.dump(weight_analysis_logs, f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# overall experiment settings
	parser.add_argument("--experiment-type", type=str, default="id_vitr")
	parser.add_argument("--experiment-name", type=str, default="hmlp-1_cond_emb-768.cc3m595k_epochs-1")
	parser.add_argument("--device", type=str, default="cuda")
	# folders and loggings
	parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
	parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
	parser.add_argument("--logs-folder", type=str, default="/home/mila/s/sparsha.mishra/projects/Hyper-Alignment/hyperalignment/logs")
	parser.add_argument("--use-wandb", type=bool, default=False)
	# model and dataset settings
	parser.add_argument("--chosen-image-encoders", type=str, default="0,1,2,3,4,5")
	parser.add_argument("--chosen-text-encoders", type=str, default="0")
	parser.add_argument("--image-embed-dim", type=int, default=768)
	parser.add_argument("--text-embed-dim", type=int, default=768)
	parser.add_argument("--hnet-cond-emb-dim", type=int, default=768)
	parser.add_argument("--use-bias", type=bool, default=True)
	parser.add_argument("--logit-scale", type=float, default=100.0)
	# datasets
	parser.add_argument("--feature-dataset", type=str, default="cc3m595k_id_vitr_raw")
	parser.add_argument("--val-dataset", type=str, default="cc3mval_id_vitr_raw")
	parser.add_argument("--test-dataset", type=str, default="mscoco_val_id_vitr_var")
	# training settings
	parser.add_argument("--scheduler", type=str, default="off")
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--eval-batch-size", type=int, default=512)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=0.0)
	parser.add_argument("--num-epochs", type=int, default=20)
	parser.add_argument("--shuffle-data", type=bool, default=True)
	parser.add_argument("--warmup-steps", type=int, default=500)
	# seeding
	parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
	parser.add_argument("--random-seed", type=int, default=0)
	# hnet ablations
	parser.add_argument("--hnet-type", type=str, default="simple")
	parser.add_argument("--lora-rank", type=int, default=32)
	parser.add_argument("--num-chunks", type=int, default=3)
	parser.add_argument("--saving", type=bool, default=True)
	parser.add_argument("--one-run", type=bool, default=False)

	args = parser.parse_args()
	seeds = [int(s) for s in args.seeds.split(",")]
	for seed in seeds:
		args.random_seed = seed
		train_joint_mapper(args)
		print("Done with seed:", seed)
