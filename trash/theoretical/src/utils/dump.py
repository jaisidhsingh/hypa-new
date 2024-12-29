
class SeparateEvaluator():
    def __init__(self, args, epochs_to_eval):
        self.experiment_name = args.experiment_name
        self.epochs_to_eval = epochs_to_eval
        self.image_embed_dim = args.image_embed_dim
        self.text_embed_dim = args.text_embed_dim
        self.eval_dataset = args.eval_dataset
        self.args = args
    
    def eval_one_epoch(self, epoch):
        ckpt_path = os.path.join(
            "../checkpoints", "naive_mapping", 
            self.experiment_name, f"ckpt_{epoch}.pt"
        )
        ckpt = torch.load(ckpt_path)["model"]
        mapper = MlpMapper(self.text_embed_dim, [], self.image_embed_dim, use_bias=True)
        mapper.load_state_dict(ckpt)

        model = CustomVLM(self.args.image_encoder, self.args.text_encoder)
        model.mapper = mapper.to(self.args.device)

        val_acc, val_loss = self.pass_over_eval_data(model)
        return val_acc, val_loss
    
    @torch.no_grad()
    def pass_over_eval_data(self, model):
        dataset = torch_datasets.CIFAR10(
            root="/workspace/datasets/cifar10",
            train=False,
            download=True,
            transform=model.image_encoder.transform
        )
        cat_texts = [f"a photo of a {c}" for c in dataset.classes]
        cat_features = model.encode_text(cat_texts)

        loader = DataLoader(dataset, batch_size=self.args.batch_size, pin_memory=True)
        bar = tqdm(total=len(loader))
        for (images, labels) in loader:
            images = images.float().to(self.args.device)
            labels = labels.long().to(self.args.device)

            image_features = model.encode_image(images)
            sim = 100 * (image_features @ cat_features.T).softmax(dim=-1)
            preds = sim.argmax(dim=-1)

            loss = F.cross_entropy(sim, labels)
            correct, total = (preds == labels).sum().item(), labels.shape[0]
            accuracy = round(correct/total * 100, 2)

            bar.set_postfix({"val_acc": accuracy, "val_loss": loss.item()})
            bar.update(1)
        
        bar.close()
        return accuracy, loss
    
    def eval_across_all_epochs(self):
        result = {}
        result["experiment_name"] = self.experiment_name
        result[f"{self.eval_dataset}_eval"] = {}

        for epoch in self.epochs_to_eval:
            val_acc, val_loss = self.eval_one_epoch(epoch)
            result[f"{self.eval_dataset}_eval"][epoch] = {"accuracy": val_acc, "loss": val_loss}
        
        return result


class JointEvaluator():
    def __init__(self, args, epochs_to_eval):
        self.experiment_name = args.experiment_name
        self.epochs_to_eval = epochs_to_eval
        self.image_embed_dim = args.image_embed_dim
        self.text_embed_dim = args.text_embed_dim
        self.eval_dataset = args.eval_dataset
        self.args = args
    
    def eval_one_epoch(self, epoch):
        ckpt_path = os.path.join(
            "../checkpoints", "joint_mapping", 
            self.experiment_name, f"ckpt_{epoch}.pt"
        )
        ckpt = torch.load(ckpt_path)["model"]
        mapper = MLP(n_in=768, n_out=768, hidden_layers=[], no_weights=True).to(self.args.device)
        mapper.eval()

        hnet = HMLP(
            mapper.param_shapes, uncond_in_size=0, 
            cond_in_size=768, layers=[],
            num_cond_embs=3
        ).to(self.args.device)
        hnet.load_state_dict(ckpt)
        hnet.eval()

        params = hnet(cond_id=2)

        model = CustomVLM(self.args.image_encoder, self.args.text_encoder)
        model.mapper = mapper.to(self.args.device)

        val_acc, val_loss = self.pass_over_eval_data(model, params)
        return val_acc, val_loss
    
    @torch.no_grad()
    def pass_over_eval_data(self, model, params):
        dataset = torch_datasets.CIFAR10(
            root="/workspace/datasets/cifar10",
            train=False,
            download=True,
            transform=model.image_encoder.transform
        )
        cat_texts = [f"a photo of a {c}" for c in dataset.classes]
        cat_features = model.encode_text_unmapped(cat_texts)
        cat_features = model.mapper(cat_features.to(self.args.device), weights=params)

        loader = DataLoader(dataset, batch_size=self.args.batch_size, pin_memory=True)
        bar = tqdm(total=len(loader))
        for (images, labels) in loader:
            images = images.float().to(self.args.device)
            labels = labels.long().to(self.args.device)

            image_features = model.encode_image(images)
            sim = 100 * (image_features @ cat_features.T).softmax(dim=-1)
            preds = sim.argmax(dim=-1)

            loss = F.cross_entropy(sim, labels)
            correct, total = (preds == labels).sum().item(), labels.shape[0]
            accuracy = round(correct/total * 100, 2)

            bar.set_postfix({"val_acc": accuracy, "val_loss": loss.item()})
            bar.update(1)
        
        bar.close()
        return accuracy, loss
    
    def eval_across_all_epochs(self):
        result = {}
        result["experiment_name"] = self.experiment_name
        result[f"{self.eval_dataset}_eval"] = {}

        for epoch in self.epochs_to_eval:
            val_acc, val_loss = self.eval_one_epoch(epoch)
            result[f"{self.eval_dataset}_eval"][epoch] = {"accuracy": val_acc, "loss": val_loss}
        
        return result


class GHN3Evaluator():
    def __init__(self, args, epochs_to_eval):
        self.experiment_name = args.experiment_name
        self.epochs_to_eval = epochs_to_eval
        self.image_embed_dim = args.image_embed_dim
        self.text_embed_dim = args.text_embed_dim
        self.eval_dataset = args.eval_dataset
        self.args = args
    
    def eval_one_epoch(self, epoch, input_image_encoders, input_text_encoders):
        ckpt_path = os.path.join(
            "../checkpoints", "joint_mapping", 
            self.experiment_name, f"ckpt_{epoch}.pt"
        )
        ckpt = torch.load(ckpt_path)
        print(ckpt.keys())
        ckpt_w, ckpt_ce = ckpt["weights"], ckpt["cond_embs"]

        mapper = MLP(n_in=768, n_out=768, hidden_layers=[], no_weights=True).to(self.args.device)
        mapper.eval()

        hnet = OracleHMLP(
            args,
            input_image_encoders, 
            input_text_encoders, 
            mapper,
            image_cond_emb_dim=384, 
            text_cond_emb_dim=384,
            image_presaved_path=None,
            variable_text_encoders=False
        )
        hnet = hnet.to(args.device)
        hnet.load_state_dict(ckpt["model"])
        hnet.eval()

        hnet_flops = get_ghn3_hypnet_flops(hnet, {"cond_input": torch.randn(3, 768).to(args.device)}, include_backward=True)
        print(hnet_flops)

        params, cond_embs = hnet.predict_weights()
        print("CE")
        print(torch.equal(cond_embs, ckpt_ce))
        print(cond_embs[0][:5])
        print(ckpt_ce[0][:5])

        print("W")
        print(torch.equal(params[0][0], ckpt_w[0][0]))
        print(params[0][0][:5])
        print(ckpt_w[0][0][:5])

        # testing only ViT rn
        params = params[0]

        model = CustomVLM(self.args.image_encoder, self.args.text_encoder)
        model.mapper = mapper.to(self.args.device)

        val_acc, val_loss = self.pass_over_eval_data(model, params)
        return val_acc, val_loss
    
    @torch.no_grad()
    def pass_over_eval_data(self, model, params):
        dataset = torch_datasets.CIFAR10(
            root="/workspace/datasets/cifar10",
            train=False,
            download=True,
            transform=model.image_encoder.transform
        )
        cat_texts = [f"a photo of a {c}" for c in dataset.classes]
        cat_features = model.encode_text_unmapped(cat_texts)
        cat_features = model.mapper(cat_features.to(self.args.device), weights=params)

        loader = DataLoader(dataset, batch_size=self.args.batch_size, pin_memory=True)
        bar = tqdm(total=len(loader))
        for (images, labels) in loader:
            images = images.float().to(self.args.device)
            labels = labels.long().to(self.args.device)

            image_features = model.encode_image(images)
            sim = 100 * (image_features @ cat_features.T).softmax(dim=-1)
            preds = sim.argmax(dim=-1)

            loss = F.cross_entropy(sim, labels)
            correct, total = (preds == labels).sum().item(), labels.shape[0]
            accuracy = round(correct/total * 100, 2)

            bar.set_postfix({"val_acc": accuracy, "val_loss": loss.item()})
            bar.update(1)
        
        bar.close()
        return accuracy, loss
    
    def eval_across_all_epochs(self):
        result = {}
        result["experiment_name"] = self.experiment_name
        result[f"{self.eval_dataset}_eval"] = {}

        input_image_encoders=["vit_base_patch16_224", "deit_base_patch16_224", "swin_small_patch4_window7_224.ms_in22k_ft_in1k"]
        input_text_encoders=["sentence-t5-base"]

        for epoch in self.epochs_to_eval:
            val_acc, val_loss = self.eval_one_epoch(epoch, input_image_encoders, input_text_encoders)
            result[f"{self.eval_dataset}_eval"][epoch] = {"accuracy": val_acc, "loss": val_loss}
        
        return result
    
    def eval_transfer_to_unseen_models(self, unseen_model_name, epoch):
        """
        Note: The hypnet needs the same number of conditional embeddings
        that were initialized during its training.

        > If you trained with N image encoders and M text encoders,
          the number of conditional embeddings should always be N*M.
        > If you want to check the alignment of 1 unseen image encoder,
          use `input_image_encoders = [new_encoder for _ in range(N)]`.
        > leave `input_text_encoders` as it was.
        """

        # OOD model, repeated for the number of ID models (to not break)
        input_image_encoders = [unseen_model_name for _ in range(3)]
        input_text_encoders = ["sentence-t5-base"]

        ckpt_path = os.path.join(
            "../checkpoints", "joint_mapping", 
            self.experiment_name, f"ckpt_{epoch}.pt"
        )
        ckpt = torch.load(ckpt_path)
        print(ckpt.keys())
        ckpt_ce = ckpt["cond_embs"]

        mapper = MLP(n_in=768, n_out=768, hidden_layers=[], no_weights=True).to(self.args.device)
        mapper.eval()

        hnet = OracleHMLP(
            args,
            input_image_encoders, 
            input_text_encoders, 
            mapper,
            image_cond_emb_dim=384, 
            text_cond_emb_dim=384,
            image_presaved_path=None,
            variable_text_encoders=False
        )
        hnet = hnet.to(args.device)
        hnet.load_state_dict(ckpt["model"])
        hnet.eval()

        hnet_flops = get_ghn3_hypnet_flops(hnet, {"cond_input": torch.randn(3, 768).to(args.device)}, include_backward=True)
        print(hnet_flops)

        params, cond_embs = hnet.predict_weights()

        # check the text cond embs
        print(torch.equal(cond_embs[0][384:], ckpt_ce[0][384:]))

        # testing one only one OOD model rn, so pickup any
        params = params[0]

        model = CustomVLM(self.args.image_encoder, self.args.text_encoder)
        model.mapper = mapper.to(self.args.device)

        val_acc, val_loss = self.pass_over_eval_data(model, params)
        return val_acc, val_loss


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="ghn3_hnet_test_2") # vitb_st5b_mscoco_1_layer_till_500
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-encoder", type=str, default="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k") # swin_small_patch4_window7_224.ms_in22k_ft_in1k
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--eval-dataset", type=str, default="cifar10")
    parser.add_argument("--image-embed-dim", type=int, default=768)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ghn3-name", type=str, default="ghn3xlm16.pt")
    parser.add_argument("--node-k", type=int, default=12)
    parser.add_argument("--node-downsampling", type=str, default="sum")
    
    args = parser.parse_args()
    return args


"""
class OldDataset(Dataset):
	def __init__(self, eval_dataset_name, transform=None):
		self.transform = transform
		self.helper = None

		if "cifar10" not in eval_dataset_name:
			self.data = torch.load(f"/workspace/datasets/{eval_dataset_name}/preprocessed_data.pt")
			self.classes = [x.lower().replace("_", " ") for x in self.data["class_list"]]
			self.image_paths = self.data["test"]["image_paths"]
			self.labels = self.data["test"]["labels"]
		else:
			if eval_dataset_name == "cifar10":
				self.helper = torch_datasets.CIFAR10(root=f"/workspace/datasets/cifar10", train=False, download=False, transform=self.transform)
			elif eval_dataset_name == "cifar100":
				self.helper = torch_datasets.CIFAR100(root=f"/workspace/datasets/cifar100", train=False, download=False, transform=self.transform)
			self.classes = self.helper.classes
	
	def __len__(self):
		return len(self.labels) if self.helper is None else len(self.helper)
	
	def __getitem__(self, idx):
		if self.helper is None:
			image_path = self.image_paths[idx]
			image = Image.open(image_path).convert("RGB")

			if self.transform is not None:
				image = self.transform(image)
			
			label = self.labels[idx]
		else:
			(image, label) = self.helper[idx]

		return image, label


class Cifar10EmbeddingClassificationDataset(Dataset):
	def __init__(self, args, text_encoder, train):
		self.image_embeddings = torch.load(f"{args.results_folder}/image_embeddings/cifar10/test/dim_{args.image_embed_dim}.pt")
		self.image_embeddings = self.image_embeddings[args.image_embed_dim][args.image_encoder]
		self.train = train

		self.helper = torch_datasets.CIFAR10(root="/workspace/datasets/cifar10", train=train, download=False, transform=None)
		self.classes = self.helper.classes

		text_encoder = text_encoder(model_name=args.text_encoder)
		self.class_embeddings = text_encoder.encode_text([f"a photo of a {c}" for c in self.classes])
		# job done, now clear from memory 
		del text_encoder

	def __len__(self):
		return self.image_embeddings.shape[0]

	def __getitem__(self, idx):
		_, label = self.helper[idx]

		if self.train:
			return self.image_embeddings[idx], self.class_embeddings[label]
		else:
			return self.image_embeddings[idx], label
"""