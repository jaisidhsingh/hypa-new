import clip
import torch
from torch.utils.data import DataLoader
import os
import sys
from copy import deepcopy
from tqdm import tqdm
from src.data import ImageCaptionDataset
from src.configs.data_configs import data_configs
from src.configs.model_configs import model_configs
from src.models import *


CKPT_FOLDER = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/id_vitr"
LOGS_FOLDER = "/home/mila/s/sparsha.mishra/projects/Hyper-Alignment/logs/id_vitr"


# @torch.no_grad()
# def get_coco_metrics(logit_scale, image_features, text_features):
#     metrics = {}

#     image_features = image_features.view(5000, 768)
#     text_features = text_features.view(5000, 5, 768)

#     logits_per_image = (logit_scale * image_features @ text_features.view(5000*5, 768).T)
#     logits_per_text = logits_per_image.T

#     # logits_per_image = logits_per_image.view(5000, 5, 5000)
#     # logits_per_text = logits_per_text.view(5000, 5, 5000)

#     logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text[:len(logits_per_image)]}
#     ground_truth = torch.arange(len(text_features)).view(-1, 1)
#     ground_truth = ground_truth.repeat(1, 5)

#     for name, logit in logits.items():
#         ranking = torch.argsort(logit, descending=True)
#         preds = torch.where(ranking == ground_truth)
#         preds = preds.numpy()
#         metrics[f"{name}_mean_rank"] = preds.mean() + 1
#         metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
#         for k in [1, 5]:
#             metrics[f"{name}_R@{k}"] = np.mean(preds < k)

#     print(metrics)
#     return metrics

# @torch.no_grad()
# def eval_coco(model, dataset, using_clip=False, device="cuda"):
#     model.mapper.eval()
#     loader = DataLoader(dataset, batch_size=128, pin_memory=False, shuffle=False, collate_fn=dataset.collate_fn)

#     image_feats, text_feats = [], []
#     bar = tqdm(total=len(loader))

#     for idx, (images, captions) in enumerate(loader):
#         B = images.shape[0]
#         images = images.float().to(device)

#         # flatten captions first (B x 5 -> B*5)
#         captions = [y for x in captions for y in x]

#         # clip tokenization
#         if using_clip:
#             captions = clip.tokenize(captions) # shape: B*5x77
#             captions = captions.to(device)

#         image_features = model.encode_image(images)
#         text_features = model.encode_text(captions)

#         # reshape text_features
#         text_features = text_features.view(B, 5, 768)

#         image_features = image_features.cpu()
#         text_features = text_features.cpu()
#         images = images.cpu()

#         # store features
#         image_feats.append(image_features.clone())
#         text_feats.append(text_features.clone())

#         del images
#         del captions
#         del image_features
#         del text_features

#         bar.update(1)

#     image_feats = torch.cat(image_feats, dim=0)
#     text_feats = torch.cat(text_feats, dim=0)
#     metrics = get_coco_metrics(model.mapper.logit_scale, image_feats, text_feats)
#     return metrics


@torch.no_grad()
def coco_captions_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0
    loss = 0

    if progress_bar:
        bar = tqdm(total=len(loader))
    logit_scale = torch.tensor(np.log(100.0)).to(device)
    for idx, (images, captions) in enumerate(loader):
        batch_size = images.shape[0]

        if using_clip:
            captions = clip.tokenize(captions).to(device)

        images = images.float().to(device)
        image_features = model.encode_image(images)

        mapped_caption_features = model.encode_text(captions)
        labels = torch.arange(batch_size, dtype=torch.long).to(device)

        sim = logit_scale * (image_features @ mapped_caption_features.T)
        preds = torch.argmax(sim, dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

        accuracy = round(correct/total * 100, 2)

        if progress_bar:
            bar.set_postfix({"accuracy": accuracy, "avg_loss": loss})

        bar.update(1)
    
    return accuracy


@torch.no_grad()
def image_classification_eval(model, loader, progress_bar=False, device="cuda"):
    correct, total = 0, 0
    loss = 0

    if progress_bar:
        bar = tqdm(total=len(loader))

    category_features = model.encode_text(
        [f"a photo of a {c}" for c in loader.dataset.classes]
    ).to(device)
    logit_scale = torch.tensor(np.log(100.0)).to(device)

    for idx, (images, labels) in enumerate(loader):
        batch_size = images.shape[0]
        labels = labels.long().to(device)

        images = images.float().to(device)
        image_features = model.encode_image(images)

        sim = logit_scale * (image_features @ category_features.T)
        preds = torch.argmax(sim, dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

        accuracy = round(correct/total * 100, 2)

        if progress_bar:
            bar.set_postfix({"accuracy": accuracy, "avg_loss": loss})

        bar.update(1)

    return accuracy


def evaluate_id_vitr_on_coco():
    get_sep_exp_name = lambda x: f"id_vitr_ie-{x}_te-0.cc3m595k_id_vitr_raw_epochs-40"
    joint_exp_name = "hmlp-1_id_vitr_ls-100.0.cc3m595k_epochs-20"

    configs = model_configs.ID_experiment_configs["id_vitr"]

    # we use mean performance across all seeds and max across all models
    separate_results = {"mscoco": {"R@1": np.zeros((6, 5, 3))}}
    joint_results = {"mscoco": {"R@1": np.zeros((6, 5, 3))}}

    separate_epochs = [1, 5, 10]
    joint_epochs = ["init", "ft-1"]

    for i in range(6): # number of models

        model = CustomVLM(configs["image_encoders"][i], configs["text_encoders"][0])
        setattr(model, "mapper", MlpMapper(768, [], 768).to("cuda"))

        kwargs = data_configs.image_caption_dataset_configs["mscoco_val"]
        kwargs["feature_dataset"] = "mscoco_val"
        kwargs["transform"] = model.image_encoder.transform
        dataset = ImageCaptionDataset(kwargs)
        loader = DataLoader(dataset, batch_size=2048, pin_memory=False, shuffle=False, collate_fn=dataset.collate_fn)

        for s in range(1): # number of seeds
            for epoch in separate_epochs:
                # run over separate mapper first
                ckpt = torch.load(os.path.join(CKPT_FOLDER, "separate", get_sep_exp_name(i), f"seed_{s}", f"ckpt_{epoch}.pt"))["model"]
                model.mapper.load_state_dict(ckpt)

                accuracy = coco_captions_eval(model, loader, using_clip=False)
                epoch_idx = separate_epochs.index(epoch)
                separate_results["mscoco"]["R@1"][i][s][epoch_idx] = accuracy

                if sys.argv[1] == "test":
                    break

            for mode in joint_epochs:
                # now run over joint mapper
                weights = []

                if mode == "init":
                    weights = torch.load(os.path.join(CKPT_FOLDER, "joint", joint_exp_name, f"seed_{s}", "ckpt_1.pt"))["mapper_params"][i]
                    model.mapper.layers[0].weight.data = weights[0]
                    model.mapper.layers[0].bias.data = weights[1]

                elif "ft" in mode:
                    ft_epoch = int(mode.split("-")[1])
                    weights = torch.load(os.path.join(CKPT_FOLDER, "joint", joint_exp_name, f"seed_{s}", f"ie-{i}_te-0", f"ft_ckpt_{ft_epoch}.pt"))["model"]
                    model.mapper.load_state_dict(weights)

                accuracy = coco_captions_eval(model, loader, using_clip=False)
                epoch_idx = joint_epochs.index(mode)
                joint_results["mscoco"]["R@1"][i][s][epoch_idx] = accuracy

                if sys.argv[1] == "test":
                    break

            if sys.argv[1] == "test":
                break

        if sys.argv[1] == "test":
            break

    results = {"separate": separate_results, "joint": joint_results}
    reduced_results = deepcopy(results)

    for mode in ["separate", "joint"]:
        reduced_results[mode]["mscoco"]["R@1"] = reduced_results[mode]["mscoco"]["R@1"].mean(1).max(0)
    
    print(reduced_results)

    torch.save(results, os.path.join(LOGS_FOLDER, "coco_eval_results.pt"))
    torch.save(reduced_results, os.path.join(LOGS_FOLDER, "coco_eval_reduced_results.pt"))
    print("All done and saved.")


if __name__ == "__main__":
    evaluate_id_vitr_on_coco()
