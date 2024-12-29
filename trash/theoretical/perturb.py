import numpy as np
import torch


def get_embedding_stats(embeddings):
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    return {"mean": mean, "std": std}


def get_perturbations_at_stats(stats, seeds, rescale_factor=1e-4):
    output = []
    mean = torch.zeros(stats["mean"].shape)
    std = torch.from_numpy(stats["std"]) * 1e-1
    for i in range(len(seeds)):
        torch.manual_seed(seeds[i])
        new_std = torch.randn(std.shape) * rescale_factor + std
        noise = torch.normal(mean, new_std)
        output.append(noise)

    return output


def perturb_embeddings_statically(embeddings, seeds, perturbations, normalize_perturbed=True):
    N, D = len(embeddings), embeddings[0].shape[0]
    output = torch.zeros((N, len(seeds)+1, D))
    output[:, 0, :] = embeddings

    for i in range(1, len(seeds)+1):
        new_embeddings = embeddings + perturbations[i-1].repeat(N, 1).to(embeddings.device)
        output[:, i, :] = new_embeddings

    if normalize_perturbed:
        output /= output.norm(dim=-1, keepdim=True)

    return output


def perturb_embeddings_dynamically(embeddings, seeds, perturbations, shuffle_seed, normalize_perturbed=True):
    N, D = len(embeddings), embeddings[0].shape[0]
    output = torch.zeros((N, len(seeds)+1, D))
    output[:, 0, :] = embeddings

    for i in range(1, len(seeds)+1):
        torch.manual_seed(shuffle_seed)
        idx = torch.randperm(perturbations[i-1].shape[-1])
        shuffled_perturbation = perturbations[i-1][idx]
        new_embeddings = embeddings + shuffled_perturbation.repeat(N, 1).to(embeddings.device)
        output[:, i, :] = new_embeddings

    if normalize_perturbed:
        output /= output.norm(dim=-1, keepdim=True)

    return output