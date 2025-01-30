import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")
import clip
import torch
from torch.utils.data import DataLoader

from models import CustomVLM, TextEncoder, ImageEncoder
from models.param_decoders import MLP
from configs.data_configs import data_configs
from configs.model_configs import model_configs
from data.image_caption_datasets import ImageCaptionDataset
from data.classification_datasets import ImageClassificationDataset


def compute_retrieval(a2b_sims, return_ranks=False):
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    report_dict = {"R@1": r1, "R@5": r5, "R@10": r10} 

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict, 0


@torch.no_grad()
def coco_captions_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0

    if progress_bar:
        bar = tqdm(total=len(loader))
    logit_scale = torch.tensor(np.log(100.0)).to(device)

    image_store = []
    text_store = []
    D_img = 0

    for idx, (images, captions) in enumerate(loader):
        batch_size = images.shape[0]
        images = images.float().to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        D_img = image_features.shape[-1]
        # image_features = image_features.repeat((5, 1))
        image_store.append(image_features)

        # average the embeddings of the 5 captions per image
        caption_store = []
        for item in captions:
            item = item[:5]  #[item[0]]
            if using_clip:
                item = clip.tokenize(item).to(device)
            item_embeddings = model.encode_text(item)
            item_embeddings /= item_embeddings.norm(dim=-1, keepdim=True)
            caption_store.append(item_embeddings.unsqueeze(0))
        text_store.append(torch.cat(caption_store, dim=0))
        
        if progress_bar:
            bar.set_postfix({"batch_index": idx})
            bar.update(1)
        
    image_store = torch.cat(image_store, dim=0).view(5000, D_img)
    text_store = torch.cat(text_store, dim=0).view(5000, 5, D_img)

    mean_recalls = {"R@1": 0, "R@5": 0, "R@10": 0}
    for i in range(1):
        sim = logit_scale * (image_store @ text_store[:, i, :].T)
        res, _ = compute_retrieval(sim.cpu().numpy())
        for k in res.keys():
            mean_recalls[k] += res[k]
    
    return mean_recalls, 0


@torch.no_grad()
def image_classification_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0

    if progress_bar:
        bar = tqdm(total=len(loader))

    logit_scale = torch.tensor(np.log(100.0)).to(device)
    class_prompt = [f"a photo of a {c}" for c in loader.dataset.classes]
    
    if using_clip:
        class_prompt = clip.tokenize(class_prompt).to(device)

    class_features = model.encode_text(class_prompt).to(device)

    running_loss = 0.0
    for idx, (images, labels) in enumerate(loader):
        batch_size = images.shape[0]
        labels = labels.long().to(device)

        if using_clip:
            captions = clip.tokenize(captions).to(device)

        images = images.float().to(device)
        image_features = model.encode_image(images)

        sim = logit_scale.exp() * image_features @ class_features.T
        preds = torch.argmax(sim, dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size
        running_loss += torch.nn.functional.cross_entropy(sim, labels).item()

        accuracy = round(correct/total * 100, 2)

        if progress_bar:
            bar.set_postfix({"accuracy": accuracy})

        bar.update(1)
    
    return accuracy, running_loss / len(loader)


def load_separate_ckpt(args, model):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints"
    path = os.path.join(folder, "ape", args.exp_name, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    ckpt = torch.load(path)["model"]
    model.load_state_dict(ckpt)
    model = model.to(args.device)
    model.eval()
    return model


def load_ood_ckpt(args, model):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper/ood_attempts"
    args.ood_results_path = args.exp_name + "_ood_ft.pt"
    print(args.ood_results_path)
    path = os.path.join(folder, args.ood_results_path)
    store = torch.load(path)
    # num_epochs = store["config"]["num_epochs"]
    [weight, bias] = store[f"epoch_1"]["mapper_params"]
    model.layers[0].weight.data = weight.to(args.device)
    model.layers[0].bias.data = bias.to(args.device)
    model = model.to(args.device)
    model.eval()
    return model


def load_mm_ckpt(args, model, vlm=False):
    folder = f"/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints/multi_mapper"   # multi_mapper
    path = os.path.join(folder, args.exp_name, f"seed_{args.seed}", f"ckpt_{args.epoch}.pt")
    chunk_size = args.num_encoders // 3 #int(args.exp_name.split("_")[1]) // 3
    
    if args.image_embed_dim == 384:
        offset = 0
    elif args.image_embed_dim == 768:
        offset = 1
    else:
        offset = 2

    index = int(offset * chunk_size) + args.encoder_index
    [weight, bias] = torch.load(path)["mapper_params"][index]
    # print(weight.shape, bias.shape)
    if not vlm:
        model.layers[0].weight.data = weight.to(args.device)
        model.layers[0].bias.data = bias.to(args.device)
    if vlm:
        model.mapper.layers[0].weight.data = weight.to(args.device)
        model.mapper.layers[0].bias.data = bias.to(args.device)
    
    model = model.to(args.device)
    model.eval()
    return model


def eval_retrieval(args, model, transform, bench):
    config = data_configs.image_caption_dataset_configs["mscoco_val"]
    config.update({"transform": transform, "feature_dataset": "mscoco_val"})

    dataset = ImageCaptionDataset(config)
    loader = DataLoader(dataset, batch_size=1024, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False)
    using_clip = args.clip_version != "off"
    recalls = coco_captions_eval(model, loader, using_clip=using_clip, device=args.device)
    return recalls, 0


def eval_classification(args, model, transform, dataset):
    root_mapping = {
        "imagenet1k": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
        "cifar10": "/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
        "cifar100": "/home/mila/s/sparsha.mishra/scratch/cifar-100-python",
    }
    kwargs = {
        "feature_dataset": dataset,
        "root": root_mapping[dataset],
        "transform": transform
    }
    dataset = ImageClassificationDataset(kwargs)
    # print(dataset.classes[0])
    loader = DataLoader(dataset, batch_size=1024, num_workers=args.num_workers, pin_memory=True)
    # using_clip = args.clip_version != "off"
    accuracy, loss = image_classification_eval(model, loader, using_clip=False, device=args.device)
    return accuracy, loss


@torch.no_grad()
def emb_eval_classification(args, model, transform, dataset):
    imagenet_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/image_embeddings/icml/eval/imagenet1k"
    path = os.path.join(imagenet_folder, f"dim_{args.image_embed_dim}", args.image_encoder, "embedded_data.pt")
    data = torch.load(path)
    
    image_features = data["inputs"].to(args.device)
    labels = data["labels"].to(args.device)
    
    root_mapping = {
        "imagenet1k": "/home/mila/s/sparsha.mishra/scratch/imagenet/val_torchvision/val",
        "cifar10": "/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision",
        "cifar100": "/home/mila/s/sparsha.mishra/scratch/cifar-100-python",
    }
    kwargs = {
        "feature_dataset": dataset,
        "root": root_mapping[dataset],
        "transform": transform
    }
    dataset = ImageClassificationDataset(kwargs)
    te = TextEncoder(args.text_encoder)
    logit_scale = torch.tensor(np.log(100.0)).to(args.device)
    class_prompt = [f"a photo of a {c}" for c in dataset.classes]
    class_features = te.encode_text(class_prompt).to(args.device)

    total = len(dataset)
    assert total == image_features.shape[0], "[ERROR]"
    mapped_features = model(class_features).to(args.device)
    mapped_features /= mapped_features.norm(dim=-1, keepdim=True)

    sim = logit_scale * (image_features @ mapped_features.T)
    corrects = (sim.argmax(dim=-1) == labels).sum().item()
    accuracy = round(corrects/total * 100, 2)
    return accuracy, 0


@torch.no_grad()
def emb_eval_retrieval(args, model, transform=None, d=None):
    image_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/image_embeddings/icml/eval/mscoco"
    image_embeddings = torch.load(os.path.join(image_folder, f"dim_{args.image_embed_dim}", args.image_encoder, "embedded_data.pt"))["image_features"]
    text_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/text_embeddings/icml/eval/mscoco"
    text_embeddings = torch.load(os.path.join(text_folder, f"dim_{args.text_embed_dim}", args.text_encoder, "embedded_data.pt"))["text_features"]
    print(image_embeddings.shape, text_embeddings.shape)

    total = image_embeddings.shape[0]
    mapped_features = model(text_embeddings.to(args.device))
    mapped_features /= mapped_features.norm(dim=-1, keepdim=True)
    
    sim = 100 * torch.einsum("bd,cnd->bnc", image_embeddings.to(args.device), mapped_features)
    acc = compute_retrieval(sim[:, 0, :].cpu().numpy())["R@1"]
    print(acc)
    # sim = 100 * image_embeddings.to(args.device) @ model(text_embeddings.to(args.device)).view(args.image_embed_dim, 5, image_embeddings.shape[0])
    # labels = torch.arange(image_embeddings.shape[0]).long().to(args.device)
    # correct = (sim.argmax(dim=1).argmax(dim=-1) == labels).sum().item()
    return acc, 0


def main(args):
    benchmark_mapping = {
        "mscoco": eval_retrieval,
        "cifar10": eval_classification,
        "cifar100": eval_classification,
        "imagenet1k": eval_classification,
    }

    benchmarks = args.benchmarks.split(",")
    metrics = {}
    result_save_file = "./eval_results.json"

    if args.clip_version == "off":
        args.image_encoder = model_configs.ID_multi_mapper_configs[args.image_embed_dim][args.encoder_index]
        model = CustomVLM(args.image_encoder, args.text_encoder)
        model.mapper = MLP(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
        
        # if args.run_type == "sep":
        #     model = load_separate_ckpt(args, model)
        # elif args.run_type == "mm":
        model = load_mm_ckpt(args, model, vlm=True)
        # elif args.run_type == "ood":
        #     model = load_ood_ckpt(args, model)
        
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, model.image_encoder.transform, bench)[0]
            metrics[bench] = metric

    else:
        model, preprocess = clip.load(args.clip_version, device=args.device)
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, preprocess, bench)
            metrics[bench] = metric 
    
    # metrics.update({"config": vars(args)})
    result = {f"epoch_{args.epoch}": metrics}
    return result


def mm_main(args):
    args.image_encoder = model_configs.ID_multi_mapper_configs[args.image_embed_dim][args.encoder_index]
    print(args.image_encoder, args.text_encoder, args.encoder_index)
    out = {"image_encoder": args.image_encoder, "seed": args.seed, "eval": {}}
    out["text_encoder"] = args.text_encoder
    
    for epoch in [1, 2, 5, 10, 20]:
        args.epoch = epoch
        benchmark_mapping = {
            "imagenet1k": emb_eval_classification,
        }

        benchmarks = ["imagenet1k"]
        metrics = {}
        
        transform = ImageEncoder(args.image_encoder).transform
        model = MLP(args.text_embed_dim, [], args.image_embed_dim).to(args.device)
        model = load_mm_ckpt(args, model)
        
        for bench in benchmarks:
            eval_fn = benchmark_mapping[bench]
            metric = eval_fn(args, model, transform, bench)[0]
            metrics[bench] = metric
        
        result = {f"epoch_{args.epoch}": metrics}
        out["eval"].update(result)
        # print(out)
     
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # main args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp-name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-type", type=str, default="sep", choices=["sep", "mm", "ood"])
    parser.add_argument("--ood-results-path", type=str, default="ood_attempt_1.pt")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--encoder-index", type=int, default=0)
    parser.add_argument("--benchmarks", type=str, default="imagenet1k")
    parser.add_argument("--clip-version", type=str, default="off")
    # model args
    parser.add_argument("--image-embed-dim", type=int, default=384)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-encoders", type=int, default=6)
    parser.add_argument("--encoder-batch", type=int, default=6)
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--image-encoder", type=str, default="vit_small_patch16_224")
    # get args
    args = parser.parse_args()

    args.exp_name = "hnet_30-10_fmlp_c-32_bs-512_lr-1e-2"
    args.encoder_index = 0
    args.image_embed_dim = 1024
    args.text_embed_dim = 768
    args.text_encoder = "sentence-t5-base"
    args.num_encoders = 30
    args.encoder_batch = 10
    args.benchmarks = "mscoco"

    args.epoch = 10
    main(args)

    # res = {}
    # for index in range(args.encoder_batch):
    #     print(index)
    #     args.encoder_index = index
    #     out = mm_main(args)
    #     out.update({"encoder_index": index})
    #     res[out["image_encoder"]] = out
    
    # print(res)
 